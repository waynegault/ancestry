import logging
import sys
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

from sqlalchemy.orm import Session

from ai.ai_interface import (
    classify_message_intent,
    extract_genealogical_entities,
    generate_genealogical_reply,
)
from core.database import (
    ConversationMetrics,
    ConversationState,
    ConversationStatusEnum,
    EngagementTracking,
    FactStatusEnum,
    FactTypeEnum,
    Person,
    SuggestedFact,
)
from core.session_manager import SessionManager
from genealogy.research_service import ResearchService
from messaging.safety import SafetyCheckResult, SafetyGuard, SafetyStatus

logger = logging.getLogger(__name__)


class InboundOrchestrator:
    """
    Orchestrates the processing of inbound messages, including safety checks,
    intent classification, and genealogical research.
    """

    def __init__(
        self,
        db_session: Session,
        research_service: ResearchService,
        session_manager: SessionManager,
    ):
        self.db = db_session
        self.research_service = research_service
        self.session_manager = session_manager
        self.safety_guard = SafetyGuard()

    def process_message(
        self,
        message_content: str,
        sender_id: str,
        conversation_id: str,
        context_history: str,
    ) -> dict[str, Any]:
        """
        Process an inbound message.

        Args:
            message_content: The content of the received message.
            sender_id: The ID of the sender (person UUID or profile_id).
            conversation_id: The ID of the conversation (unused for state lookup, but logged).
            context_history: A string representation of the conversation history
                             (formatted for AI context).

        Returns:
            A dictionary containing the processing results.
        """
        logger.info(f"Processing inbound message from {sender_id} (Conv: {conversation_id})")

        # 0. Resolve Person
        person = self._resolve_person(sender_id)
        if not person:
            logger.error(f"Could not resolve person for sender_id: {sender_id}")
            return {"status": "error", "reason": "person_not_found"}

        # 1. Safety Check
        safety_result = self.safety_guard.check_message(message_content)
        if safety_result.status != SafetyStatus.SAFE:
            # Opt-out detection is EXPECTED behavior - log at appropriate level
            if safety_result.status == SafetyStatus.OPT_OUT:
                logger.info(f"Opt-out detected for {sender_id}: skipping automated response ({safety_result.reason})")
            else:
                logger.warning(f"Safety check flagged message from {sender_id}: {safety_result.reason}")
            self._handle_unsafe_message(person, safety_result)
            return {
                "status": "unsafe",
                "safety_result": safety_result,
                "action": "flagged",
            }

        # 2. Intent Classification
        intent = classify_message_intent(context_history, self.session_manager)
        logger.info(f"Classified intent for {sender_id}: {intent}")

        # Update Conversation State
        self._update_conversation_state(person, intent)

        # 3. Genealogical Research & Reply Generation
        research_results = None
        generated_reply = None
        extracted_data = None

        if intent in {"PRODUCTIVE", "ENTHUSIASTIC"}:
            # A. Extract Entities
            extracted_data = extract_genealogical_entities(context_history, self.session_manager)

            # B. Harvest Facts
            if extracted_data:
                self._harvest_facts(person, extracted_data)

            # C. Research & Reply
            if extracted_data:
                research_results = self._perform_genealogical_research(extracted_data)

                # D. Generate Reply
                if research_results:
                    # Format data for AI
                    genealogical_data_str = self._format_research_results(research_results)
                    generated_reply = generate_genealogical_reply(
                        context_history, message_content, genealogical_data_str, self.session_manager
                    )

        # 4. Update Metrics
        self._update_metrics(person, intent, generated_reply is not None, extracted_data is not None)

        return {
            "status": "processed",
            "intent": intent,
            "safety_result": safety_result,
            "research_results": research_results,
            "generated_reply": generated_reply,
            "extracted_data": extracted_data,
        }

    def _resolve_person(self, sender_id: str) -> Optional[Person]:
        """Resolve Person object from sender_id (UUID or profile_id)."""
        return (
            self.db.query(Person).filter((Person.profile_id == sender_id) | (Person.uuid == sender_id.upper())).first()
        )

    _FACT_KEYWORD_MAP: ClassVar[dict[FactTypeEnum, tuple[str, ...]]] = {
        FactTypeEnum.BIRTH: ("born", "birth"),
        FactTypeEnum.DEATH: ("died", "death", "passed away"),
        FactTypeEnum.MARRIAGE: ("married", "marriage", "spouse", "husband", "wife"),
        FactTypeEnum.LOCATION: ("lived", "resided", "location", "place"),
        FactTypeEnum.RELATIONSHIP: ("relationship", "cousin", "aunt", "uncle", "grandparent"),
    }

    @classmethod
    def _infer_fact_type(cls, text: str) -> FactTypeEnum:
        """Infer FactTypeEnum from text content."""
        text_lower = text.lower()
        for fact_type, keywords in cls._FACT_KEYWORD_MAP.items():
            if any(kw in text_lower for kw in keywords):
                return fact_type
        return FactTypeEnum.OTHER

    def _harvest_facts(self, person: Person, extracted_data: dict[str, Any]) -> None:
        """Create SuggestedFact records from extracted data."""
        data = extracted_data.get("extracted_data", {})

        # Process mentioned people as potential facts
        for mentioned_person in data.get("mentioned_people", []):
            fact_desc = f"Mentioned Person: {mentioned_person.get('name')}"
            details: list[str] = []
            if mentioned_person.get('birth_year'):
                details.append(f"Born: {mentioned_person['birth_year']}")
            if mentioned_person.get('birth_place'):
                details.append(f"Birth Place: {mentioned_person['birth_place']}")
            if mentioned_person.get('death_year'):
                details.append(f"Died: {mentioned_person['death_year']}")
            if mentioned_person.get('relationship'):
                details.append(f"Relationship: {mentioned_person['relationship']}")

            if details:
                fact_desc += " (" + ", ".join(details) + ")"

            fact = SuggestedFact(
                people_id=person.id,
                fact_type=self._infer_fact_type(fact_desc),
                new_value=fact_desc,
                confidence_score=80,  # Placeholder
                status=FactStatusEnum.PENDING,
            )
            self.db.add(fact)

        # Process specific key facts
        for key_fact in data.get("key_facts", []):
            fact = SuggestedFact(
                people_id=person.id,
                fact_type=self._infer_fact_type(key_fact),
                new_value=key_fact,
                confidence_score=80,  # Placeholder
                status=FactStatusEnum.PENDING,
            )
            self.db.add(fact)

        self.db.commit()

    def _perform_genealogical_research(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Perform research based on extracted entities.
        Returns a list of matches/findings.
        """
        results: list[dict[str, Any]] = []
        data = extracted_data.get("extracted_data", {})

        # Search for mentioned people in our tree
        for mentioned_person in data.get("mentioned_people", []):
            name = mentioned_person.get("name")
            if not name:
                continue

            # Construct search criteria
            criteria = {"full_name": name}
            if mentioned_person.get("birth_year"):
                criteria["birth_year"] = mentioned_person["birth_year"]

            # Use ResearchService to search
            # Note: ResearchService.search_people expects specific dict structure
            # We map simple criteria to what search_people expects
            search_criteria = {
                "first_name": mentioned_person.get("first_name"),
                "surname": mentioned_person.get("last_name"),
                "birth_year": mentioned_person.get("birth_year"),
                "birth_place": mentioned_person.get("birth_place"),
                "death_place": mentioned_person.get("death_place"),
            }

            # Remove None values
            search_criteria = {k: v for k, v in search_criteria.items() if v}

            if not search_criteria:
                continue

            # Default scoring weights/flexibility (could be config driven)
            matches = self.research_service.search_people(
                filter_criteria=search_criteria,
                scoring_criteria=search_criteria,
                scoring_weights={"name_match": 50, "date_match": 30, "place_match": 20},
                date_flex={"year_match_range": 2},
            )

            if matches:
                top_match = matches[0]
                # If we have a good match, get relationship path
                path = self.research_service.get_relationship_path(
                    start_id="ROOT",  # Assuming ROOT is the user
                    end_id=top_match["id"],
                )

                results.append({"query": mentioned_person, "match": top_match, "relationship_path": path})

        return results

    @staticmethod
    def _format_research_results(results: list[dict[str, Any]]) -> str:
        """Format research results for the AI prompt."""
        output: list[str] = []
        for item in results:
            match = item["match"]
            path = item.get("relationship_path")

            entry = f"Found in tree: {match['full_name_disp']} (ID: {match['display_id']})\n"
            entry += f"  - Birth: {match.get('birth_date', 'Unknown')} in {match.get('birth_place', 'Unknown')}\n"
            entry += f"  - Death: {match.get('death_date', 'Unknown')} in {match.get('death_place', 'Unknown')}\n"

            if path:
                entry += "  - Relationship Path:\n"
                for step in path:
                    entry += f"    -> {step.get('name')} ({step.get('relationship')})\n"
            else:
                entry += "  - Relationship Path: Not found or direct calculation failed.\n"

            output.append(entry)

        if not output:
            return "No matching records found in the family tree."

        return "\n".join(output)

    def _handle_unsafe_message(self, person: Person, safety_result: SafetyCheckResult) -> None:
        """Handle a message that failed the safety check."""
        # Update conversation state to flagged
        state = self._get_or_create_conversation_state(person)
        if state:
            state.safety_flag = True
            # Append reason to ai_summary
            current_summary = state.ai_summary or ""
            state.ai_summary = f"SAFETY FLAG: {safety_result.reason}\n{current_summary}"
            state.status = ConversationStatusEnum.HUMAN_REVIEW
            self.db.commit()

    def _update_conversation_state(self, person: Person, intent: Optional[str]) -> None:
        """Update the conversation state in the database."""
        state = self._get_or_create_conversation_state(person)
        if state:
            state.last_intent = intent

            # Update status based on intent
            if intent == "DESIST":
                state.status = ConversationStatusEnum.OPT_OUT
            elif intent == "PRODUCTIVE":
                state.status = ConversationStatusEnum.ACTIVE

            self.db.commit()

    def _get_or_create_conversation_state(self, person: Person) -> Optional[ConversationState]:
        """Get existing conversation state or create a new one."""
        state = self.db.query(ConversationState).filter(ConversationState.people_id == person.id).first()
        if not state:
            state = ConversationState(
                people_id=person.id,
                status=ConversationStatusEnum.ACTIVE,
            )
            self.db.add(state)
            self.db.commit()
        return state

    def _update_metrics(
        self, person: Person, intent: Optional[str], reply_generated: bool, facts_extracted: bool
    ) -> None:
        """Update conversation metrics and track engagement events."""
        # Get or create metrics record
        metrics = self.db.query(ConversationMetrics).filter(ConversationMetrics.people_id == person.id).first()
        if not metrics:
            metrics = ConversationMetrics(people_id=person.id)
            self.db.add(metrics)

        # Update basic counts
        metrics.messages_received += 1
        metrics.last_message_received = datetime.now(timezone.utc)

        if not metrics.first_response_received:
            metrics.first_response_received = True
            metrics.first_response_date = metrics.last_message_received
            # Calculate time to first response if first_message_sent exists
            if metrics.first_message_sent is not None and metrics.first_response_date is not None:
                # Pylance doesn't infer that these are not None inside the if block for SQLAlchemy models sometimes
                sent_time = metrics.first_message_sent
                resp_time = metrics.first_response_date
                if sent_time and resp_time:
                    delta = resp_time - sent_time
                    metrics.time_to_first_response_hours = delta.total_seconds() / 3600.0

        # Track Engagement Event
        event = EngagementTracking(
            people_id=person.id,
            event_type="message_received",
            event_description=f"Received message with intent: {intent}",
            event_data=f'{{"intent": "{intent}", "reply_generated": {reply_generated}, "facts_extracted": {facts_extracted}}}',
            conversation_phase=metrics.conversation_phase,
        )
        self.db.add(event)

        if reply_generated:
            metrics.messages_sent += 1  # Assuming we send it immediately or count it as generated
            # Track reply event
            reply_event = EngagementTracking(
                people_id=person.id,
                event_type="reply_generated",
                event_description="AI generated genealogical reply",
                conversation_phase=metrics.conversation_phase,
            )
            self.db.add(reply_event)

        if facts_extracted:
            metrics.research_tasks_created += 1  # Using this field for extracted facts count for now
            # Track extraction event
            extract_event = EngagementTracking(
                people_id=person.id,
                event_type="facts_extracted",
                event_description="Genealogical facts extracted from message",
                conversation_phase=metrics.conversation_phase,
            )
            self.db.add(extract_event)

        self.db.commit()


# ==============================================================================
# MODULE TESTS
# ==============================================================================


def module_tests() -> bool:
    """
    Run comprehensive unit tests for the InboundOrchestrator.
    Uses the separate test module messaging/test_inbound.py.
    """
    import sys
    import unittest

    from testing.test_framework import Colors, Icons, TestSuite

    # Initialize the standard test suite for reporting
    suite = TestSuite("Inbound Orchestrator", "messaging/inbound.py")
    suite.start_suite()

    print(f"{Colors.BLUE}{Icons.GEAR} Running unittest suite from messaging/test_inbound.py...{Colors.RESET}")

    # Load tests from the separate test file
    try:
        from messaging import test_inbound

        # Create a unittest suite
        loader = unittest.TestLoader()
        unittest_suite = loader.loadTestsFromModule(test_inbound)

        # Run the tests using a custom runner or just TextTestRunner
        # We use TextTestRunner but capture the result to update our TestSuite stats
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
        result = runner.run(unittest_suite)

        # Update the custom suite stats based on unittest results
        suite.tests_run = result.testsRun
        suite.tests_failed = len(result.failures) + len(result.errors)
        suite.tests_passed = suite.tests_run - suite.tests_failed

        # We don't have individual test details for the custom suite report,
        # but we can report the aggregate result.

        if result.wasSuccessful():
            print(f"\n{Colors.GREEN}{Icons.PASS} All unit tests passed.{Colors.RESET}")
            return True
        print(f"\n{Colors.RED}{Icons.FAIL} Unit tests failed.{Colors.RESET}")
        return False

    except ImportError as e:
        print(f"{Colors.RED}Failed to import test module: {e}{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}An error occurred during testing: {e}{Colors.RESET}")
        return False


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
