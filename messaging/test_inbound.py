"""Test suite for InboundOrchestrator validation.

Validates the complete inbound messaging flow including SMS processing,
message categorization, database interactions, and response orchestration.
Includes tests for spam filtering, opt-out processing, and error handling.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from core.database import (
    ConversationMetrics,
    ConversationState,
    ConversationStatusEnum,
    Person,
    SuggestedFact,
)
from messaging.inbound import InboundOrchestrator
from messaging.safety import SafetyCheckResult, SafetyStatus


class TestInboundOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
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

        self.patcher_semantic = patch('messaging.inbound.SemanticSearchService')
        self.mock_semantic_cls = self.patcher_semantic.start()
        self.mock_semantic = self.mock_semantic_cls.return_value
        self.mock_semantic.should_run.return_value = False

        self.orchestrator = InboundOrchestrator(self.mock_db, self.mock_research_service, self.mock_session_manager)

    def tearDown(self) -> None:
        self.patcher_safety.stop()
        self.patcher_classify.stop()
        self.patcher_extract.stop()
        self.patcher_generate.stop()
        self.patcher_semantic.stop()

    def _setup_db_mocks(self, person_found: bool = True, state_found: bool = True) -> tuple[Any, Any]:
        mock_person = MagicMock()
        mock_person.id = 123
        mock_person.profile_id = 'sender1'
        mock_person.display_name = 'Sender One'
        mock_person.birth_year = None
        mock_person.death_year = None

        mock_state = MagicMock(spec=ConversationState)
        mock_state.status = ConversationStatusEnum.ACTIVE

        # Add transition_status method that updates the status
        def transition_status(
            new_status: ConversationStatusEnum,
            reason: str = "",  # noqa: ARG001
            triggered_by: str = "test",  # noqa: ARG001
        ) -> bool:
            if mock_state.status != new_status:
                mock_state.status = new_status
                return True
            return False

        mock_state.transition_status = transition_status

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
            elif model == SuggestedFact:
                mock_query.filter.return_value.all.return_value = []
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
        self.assertIsNone(result.get('semantic_search'))

        # Verify DB interactions
        # Should add SuggestedFact (2 times: person and key fact)
        # Should add EngagementTracking (3 times: received, reply, extracted)
        self.assertTrue(self.mock_db.add.called)
        # We can't easily count exact calls because add is called for state, metrics, facts, events
        # But we can check if SuggestedFact was added

        # Check if research service was called
        self.mock_research_service.search_people.assert_called()
        self.mock_research_service.get_relationship_path.assert_called()

    def test_process_message_safe_productive_semantic_search(self):
        # Setup
        self.mock_safety_guard.check_message.return_value = SafetyCheckResult(
            status=SafetyStatus.SAFE, reason='', flagged_terms=[]
        )
        self.mock_classify.return_value = 'PRODUCTIVE'

        self.mock_extract.return_value = {
            'extracted_data': {
                'mentioned_people': [{'name': 'John Doe', 'birth_year': 1900}],
                'key_facts': ['Lived in London'],
            }
        }

        self.mock_semantic.should_run.return_value = True
        semantic_result = MagicMock()
        semantic_result.to_dict.return_value = {'intent': 'question', 'final_answer': 'Test answer'}
        self.mock_semantic.search.return_value = semantic_result

        self.mock_generate.return_value = 'Hello, I found John Doe in my tree.'
        _, _ = self._setup_db_mocks()

        # Execute
        result = self.orchestrator.process_message('Who is John Doe?', 'sender1', 'conv1', 'history')

        # Verify
        self.assertEqual(result['status'], 'processed')
        self.assertEqual(result['intent'], 'PRODUCTIVE')
        self.assertEqual(result['semantic_search'], {'intent': 'question', 'final_answer': 'Test answer'})
        self.mock_semantic.search.assert_called_once()
        self.mock_semantic.persist_jsonl.assert_called_once()

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


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner


def run_tests() -> bool:
    """Run tests using the standardized TestSuite format."""
    suite = TestSuite("Inbound Orchestrator Tests", "messaging/test_inbound.py")
    suite.start_suite()

    # Load the unittest suite
    loader = unittest.TestLoader()
    unittest_suite = loader.loadTestsFromTestCase(TestInboundOrchestrator)

    # Bridge unittest to TestSuite
    # We iterate through the tests in the unittest suite and run them via our TestSuite
    for test in unittest_suite:
        # Each 'test' is an instance of TestInboundOrchestrator with a specific test method
        test_method_name = str(test).split(' ')[0]  # Extract method name from string representation

        # Create a closure to capture the current test instance
        from typing import Callable, Union

        def make_run_adapter(current_test: Union[unittest.TestCase, unittest.TestSuite]) -> Callable[[], None]:
            def run_adapter() -> None:
                # Create a fresh instance for each test run
                result = unittest.TestResult()
                current_test.run(result)

                if not result.wasSuccessful():
                    if result.errors:
                        raise Exception(result.errors[0][1])
                    if result.failures:
                        raise AssertionError(result.failures[0][1])

            return run_adapter

        suite.run_test(test_method_name, make_run_adapter(test), f"Run {test_method_name} from unittest suite")

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(run_tests)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
