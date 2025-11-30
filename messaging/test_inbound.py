import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from database import (
    ConversationMetrics,
    ConversationState,
    ConversationStatusEnum,
    Person,
)
from messaging.inbound import InboundOrchestrator
from messaging.safety import SafetyCheckResult, SafetyStatus


class TestInboundOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_research_service = MagicMock()
        self.mock_session_manager = MagicMock()

        # Mock SafetyGuard
        self.patcher_safety = patch('messaging.inbound.SafetyGuard')
        self.mock_safety_guard_cls = self.patcher_safety.start()
        self.mock_safety_guard = self.mock_safety_guard_cls.return_value

        # Mock AI functions
        self.patcher_classify = patch('messaging.inbound.classify_message_intent')
        self.mock_classify = self.patcher_classify.start()

        self.patcher_extract = patch('messaging.inbound.extract_genealogical_entities')
        self.mock_extract = self.patcher_extract.start()

        self.patcher_generate = patch('messaging.inbound.generate_genealogical_reply')
        self.mock_generate = self.patcher_generate.start()

        self.orchestrator = InboundOrchestrator(self.mock_db, self.mock_research_service, self.mock_session_manager)

    def tearDown(self):
        self.patcher_safety.stop()
        self.patcher_classify.stop()
        self.patcher_extract.stop()
        self.patcher_generate.stop()

    def _setup_db_mocks(self, person_found: bool = True, state_found: bool = True):
        mock_person = MagicMock()
        mock_person.id = 123
        mock_person.profile_id = 'sender1'

        mock_state = MagicMock(spec=ConversationState)
        mock_metrics = MagicMock(spec=ConversationMetrics)
        mock_metrics.messages_received = 0
        mock_metrics.first_response_received = False

        def query_side_effect(model: Any):
            mock_query = MagicMock()
            if model == Person:
                mock_query.filter.return_value.first.return_value = mock_person if person_found else None
            elif model == ConversationState:
                mock_query.filter.return_value.first.return_value = mock_state if state_found else None
            elif model == ConversationMetrics:
                mock_query.filter.return_value.first.return_value = mock_metrics
            return mock_query

        self.mock_db.query.side_effect = query_side_effect
        return mock_person, mock_state

    def test_process_message_unsafe(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.UNSAFE, reason='Hostility', flagged_terms=['hate']
        )
        self._setup_db_mocks()

        # Execute
        result = self.orchestrator.process_message('I hate you', 'sender1', 'conv1', 'history')

        # Verify
        self.assertEqual(result['status'], 'unsafe')
        self.assertEqual(result['action'], 'flagged')
        self.mock_db.commit.assert_called()  # Should commit state change

    def test_process_message_safe_productive_rag(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.SAFE, reason='', flagged_terms=[]
        )
        self.mock_classify.return_value = 'PRODUCTIVE'

        # Mock Extraction
        self.mock_extract.return_value = {
            'extracted_data': {
                'mentioned_people': [{'name': 'John Doe', 'birth_year': 1900}],
                'key_facts': ['Lived in London'],
            }
        }

        # Mock Research
        self.mock_research_service.search_people.return_value = [
            {'id': 'P1', 'full_name_disp': 'John Doe', 'display_id': 'P1'}
        ]
        self.mock_research_service.get_relationship_path.return_value = [
            {'name': 'John Doe', 'relationship': 'Grandfather'}
        ]

        # Mock Generation
        self.mock_generate.return_value = 'Hello, I found John Doe in my tree.'

        _, _ = self._setup_db_mocks()

        # Execute
        result = self.orchestrator.process_message('My grandfather was John Doe', 'sender1', 'conv1', 'history')

        # Verify
        self.assertEqual(result['status'], 'processed')
        self.assertEqual(result['intent'], 'PRODUCTIVE')
        self.assertEqual(result['generated_reply'], 'Hello, I found John Doe in my tree.')

        # Verify DB interactions
        # Should add SuggestedFact (2 times: person and key fact)
        # Should add EngagementTracking (3 times: received, reply, extracted)
        self.assertTrue(self.mock_db.add.called)
        # We can't easily count exact calls because add is called for state, metrics, facts, events
        # But we can check if SuggestedFact was added

        # Check if research service was called
        self.mock_research_service.search_people.assert_called()
        self.mock_research_service.get_relationship_path.assert_called()

    def test_process_message_desist(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.SAFE, reason='', flagged_terms=[]
        )
        self.mock_classify.return_value = 'DESIST'

        _, mock_state = self._setup_db_mocks()

        # Execute
        result = self.orchestrator.process_message('Stop messaging me', 'sender1', 'conv1', 'history')

        # Verify
        self.assertEqual(result['intent'], 'DESIST')
        self.assertEqual(mock_state.status, ConversationStatusEnum.OPT_OUT)

    def test_process_message_person_not_found(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.SAFE, reason='', flagged_terms=[]
        )
        self._setup_db_mocks(person_found=False)

        # Execute
        result = self.orchestrator.process_message('Hello', 'sender1', 'conv1', 'history')

        # Verify
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['reason'], 'person_not_found')
