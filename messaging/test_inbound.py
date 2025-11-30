import unittest
from unittest.mock import MagicMock, patch

from database import ConversationState, ConversationStatusEnum, Person
from messaging.inbound import InboundOrchestrator
from messaging.safety import SafetyCheckResult, SafetyStatus


class TestInboundOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_research_service = MagicMock()
        self.mock_session_manager = MagicMock()

        # Mock SafetyGuard
        self.patcher_safety = patch("messaging.inbound.SafetyGuard")
        self.mock_safety_guard_cls = self.patcher_safety.start()
        self.mock_safety_guard = self.mock_safety_guard_cls.return_value

        # Mock classify_message_intent
        self.patcher_classify = patch("messaging.inbound.classify_message_intent")
        self.mock_classify = self.patcher_classify.start()

        self.orchestrator = InboundOrchestrator(self.mock_db, self.mock_research_service, self.mock_session_manager)

    def tearDown(self):
        self.patcher_safety.stop()
        self.patcher_classify.stop()

    def _setup_db_mocks(self, person_found=True, state_found=True):
        mock_person = MagicMock()
        mock_person.id = 123

        mock_state = MagicMock(spec=ConversationState)

        def query_side_effect(model):
            mock_query = MagicMock()
            if model == Person:
                mock_query.filter.return_value.first.return_value = mock_person if person_found else None
            elif model == ConversationState:
                mock_query.filter.return_value.first.return_value = mock_state if state_found else None
            return mock_query

        self.mock_db.query.side_effect = query_side_effect
        return mock_person, mock_state

    def test_process_message_unsafe(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.UNSAFE, reason="Hostility", flagged_terms=["hate"]
        )
        self._setup_db_mocks()

        # Execute
        result = self.orchestrator.process_message("I hate you", "sender1", "conv1", "history")

        # Verify
        self.assertEqual(result["status"], "unsafe")
        self.assertEqual(result["action"], "flagged")
        self.mock_db.commit.assert_called()  # Should commit state change

    def test_process_message_safe_productive(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.SAFE, reason="", flagged_terms=[]
        )
        self.mock_classify.return_value = "PRODUCTIVE"

        _, mock_state = self._setup_db_mocks()

        # Execute
        result = self.orchestrator.process_message("Here is my info", "sender1", "conv1", "history")

        # Verify
        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["intent"], "PRODUCTIVE")
        self.assertEqual(mock_state.status, ConversationStatusEnum.ACTIVE)
        self.mock_db.commit.assert_called()

    def test_process_message_desist(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.SAFE, reason="", flagged_terms=[]
        )
        self.mock_classify.return_value = "DESIST"

        _, mock_state = self._setup_db_mocks()

        # Execute
        result = self.orchestrator.process_message("Stop messaging me", "sender1", "conv1", "history")

        # Verify
        self.assertEqual(result["intent"], "DESIST")
        self.assertEqual(mock_state.status, ConversationStatusEnum.OPT_OUT)

    def test_process_message_person_not_found(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.SAFE, reason="", flagged_terms=[]
        )
        self.mock_classify.return_value = "PRODUCTIVE"

        self._setup_db_mocks(person_found=False)

        # Execute
        result = self.orchestrator.process_message("Hello", "unknown_sender", "conv1", "history")

        # Verify
        # Should still process but not update state (and log warning)
        self.assertEqual(result["status"], "processed")
        self.mock_db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
