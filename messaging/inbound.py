import logging
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.orm import Session

from ai.ai_interface import (
    classify_message_intent,
    extract_genealogical_entities,
    generate_genealogical_reply,
)
from config import config_schema
from core.database import (
    ConflictStatusEnum,
    ConversationLog,
    ConversationMetrics,
    ConversationState,
    ConversationStatusEnum,
    DataConflict,
    EngagementTracking,
    FactStatusEnum,
    FactTypeEnum,
    MessageDirectionEnum,
    Person,
    PersonStatusEnum,
    SuggestedFact,
)
from core.session_manager import SessionManager
from genealogy.fact_validator import ConflictType, ExtractedFact, FactValidator, extract_facts_from_ai_response
from genealogy.research_service import ResearchService
from genealogy.semantic_search import SemanticSearchService
from messaging.safety import SafetyCheckResult, SafetyGuard, SafetyStatus

logger = logging.getLogger(__name__)


def _record_messaging_counter(metric_name: str, *, labels: dict[str, str]) -> None:
    """Best-effort internal metrics emission (safe for offline tests)."""
    try:
        from core.metrics_collector import get_metrics_registry

        get_metrics_registry().record_metric("Messaging", metric_name, 1.0, labels)
    except Exception:
        return


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

        inbound_log = self._persist_inbound_conversation_log(person, conversation_id, message_content)

        # 1. Phase 2 Critical Alert Check (must run before any AI work)
        critical_result = self.safety_guard.check_critical_alerts(message_content)
        if critical_result.status == SafetyStatus.CRITICAL_ALERT:
            _record_messaging_counter(
                "sends_blocked",
                labels={
                    "source": "inbound",
                    "reason": "critical_alert",
                    "category": str(getattr(critical_result.category, "value", "unknown")),
                },
            )
            logger.error(
                "Critical alert detected for %s (Conv: %s): %s", sender_id, conversation_id, critical_result.reason
            )
            self._handle_unsafe_message(person, critical_result)
            return {
                "status": "unsafe",
                "safety_result": critical_result,
                "action": "human_review",
            }

        if critical_result.status == SafetyStatus.HIGH_VALUE:
            _record_messaging_counter(
                "high_value_discovery",
                labels={
                    "source": "inbound",
                    "category": str(getattr(critical_result.category, "value", "unknown")),
                },
            )
            # High-value is notification-only: keep processing, but annotate state for review.
            with suppress(Exception):
                state = self._get_or_create_conversation_state(person)
                if state is not None:
                    current_summary = state.ai_summary or ""
                    note = f"HIGH_VALUE_DISCOVERY: {critical_result.reason} | Terms={critical_result.flagged_terms}\n"
                    state.ai_summary = note + current_summary
                    self.db.commit()

        # 2. Legacy Safety Check (opt-out / danger / hostility)
        safety_result = self.safety_guard.check_message(message_content)
        if safety_result.status != SafetyStatus.SAFE:
            if safety_result.status == SafetyStatus.OPT_OUT:
                _record_messaging_counter(
                    "sends_blocked",
                    labels={"source": "inbound", "reason": "safety_opt_out"},
                )
            else:
                _record_messaging_counter(
                    "sends_blocked",
                    labels={"source": "inbound", "reason": "safety_unsafe"},
                )
            # Opt-out detection is EXPECTED behavior - log at appropriate level
            if safety_result.status == SafetyStatus.OPT_OUT:
                logger.info(f"Opt-out detected for {sender_id}: skipping automated response ({safety_result.reason})")
            else:
                logger.warning(f"Safety check flagged message from {sender_id}: {safety_result.reason}")
            self._handle_unsafe_message(person, safety_result)
            return {
                "status": "unsafe",
                "safety_result": safety_result,
                "action": "opt_out" if safety_result.status == SafetyStatus.OPT_OUT else "flagged",
            }

        # 3. Intent Classification
        intent = classify_message_intent(context_history, self.session_manager)
        logger.info(f"Classified intent for {sender_id}: {intent}")

        self._attach_intent_to_log(inbound_log, intent)

        # Update Conversation State
        self._update_conversation_state(person, intent)

        source_message_id = getattr(inbound_log, "id", None) if inbound_log is not None else None

        research_results, generated_reply, extracted_data, semantic_search = self._run_research_flow(
            intent=intent,
            person=person,
            message_content=message_content,
            sender_id=sender_id,
            conversation_id=conversation_id,
            context_history=context_history,
            source_message_id=source_message_id,
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
            "semantic_search": semantic_search,
        }

    @staticmethod
    def _get_message_truncation_length() -> int:
        try:
            return int(getattr(config_schema, "message_truncation_length", 500))
        except Exception:
            return 500

    def _persist_inbound_conversation_log(
        self, person: Person, conversation_id: str, message_content: str
    ) -> Optional[ConversationLog]:
        log_entry: Optional[ConversationLog] = None
        max_len = self._get_message_truncation_length()

        try:
            log_entry = ConversationLog(
                conversation_id=conversation_id,
                direction=MessageDirectionEnum.IN,
                people_id=person.id,
                latest_message_content=(message_content or "")[:max_len],
                latest_timestamp=datetime.now(timezone.utc),
                ai_sentiment=None,
            )
            self.db.add(log_entry)
            self.db.flush()
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to persist inbound ConversationLog (non-fatal): %s", exc)
            return None

        return log_entry

    def _persist_outbound_generated_reply_log(
        self,
        person: Person,
        conversation_id: str,
        reply_content: str,
        *,
        intent: Optional[str],
        research_matches_count: int,
        semantic_search_ran: bool,
    ) -> Optional[ConversationLog]:
        log_entry: Optional[ConversationLog] = None
        max_len = self._get_message_truncation_length()

        template_key = "INBOUND_GENERATED_REPLY"
        intent_phrase = intent or "Unknown"
        semantic_phrase = "with semantic search" if semantic_search_ran else "without semantic search"
        template_reason = (
            f"Inbound AI reply generated after intent '{intent_phrase}' using {research_matches_count} "
            f"research match(es), {semantic_phrase}"
        )
        enhanced_status = f"generated_reply | Template: {template_key} ({template_reason})"

        try:
            log_entry = ConversationLog(
                conversation_id=conversation_id,
                direction=MessageDirectionEnum.OUT,
                people_id=person.id,
                latest_message_content=(reply_content or "")[:max_len],
                latest_timestamp=datetime.now(timezone.utc),
                ai_sentiment=None,
                message_template_id=None,
                script_message_status=enhanced_status,
            )
            self.db.add(log_entry)
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to persist outbound generated-reply ConversationLog (non-fatal): %s", exc)
            return None

        return log_entry

    @staticmethod
    def _attach_intent_to_log(inbound_log: Optional[ConversationLog], intent: Optional[str]) -> None:
        if inbound_log is None:
            return
        with suppress(Exception):
            inbound_log.ai_sentiment = intent

    @staticmethod
    def _maybe_run_semantic_search(
        *,
        message_content: str,
        sender_id: str,
        conversation_id: str,
        person: Person,
        extracted_data: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        semantic_service = SemanticSearchService()
        if not semantic_service.should_run(message_content):
            return None

        try:
            semantic_result = semantic_service.search(
                message_content,
                extracted_entities=extracted_data,
            )
            semantic_search = semantic_result.to_dict()
            semantic_service.persist_jsonl(
                payload=semantic_search,
                person_id=getattr(person, "id", None),
                sender_id=sender_id,
                conversation_id=conversation_id,
            )
            return semantic_search
        except Exception as exc:  # pragma: no cover
            logger.debug("Semantic search failed (non-fatal): %s", exc)
            return None

    def _run_research_flow(
        self,
        *,
        intent: Optional[str],
        person: Person,
        message_content: str,
        sender_id: str,
        conversation_id: str,
        context_history: str,
        source_message_id: Optional[int],
    ) -> tuple[Optional[list[dict[str, Any]]], Optional[str], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        if intent not in {"PRODUCTIVE", "ENTHUSIASTIC"}:
            return None, None, None, None

        extracted_data = extract_genealogical_entities(context_history, self.session_manager)
        semantic_search = self._maybe_run_semantic_search(
            message_content=message_content,
            sender_id=sender_id,
            conversation_id=conversation_id,
            person=person,
            extracted_data=extracted_data,
        )

        if extracted_data:
            self._harvest_facts(
                person,
                extracted_data,
                original_message=message_content,
                conversation_id=conversation_id,
                source_message_id=source_message_id,
            )

        research_results: Optional[list[dict[str, Any]]] = None
        generated_reply: Optional[str] = None

        if extracted_data:
            research_results = self._perform_genealogical_research(extracted_data)
            if research_results:
                genealogical_data_str = self._format_research_results(research_results)

                tree_lookup_results = ""
                relationship_context = ""
                try:
                    match_uuid = getattr(person, "uuid", None)
                    if isinstance(match_uuid, str) and match_uuid:
                        from ai.context_builder import ContextBuilder

                        match_context = ContextBuilder(db_session=self.db).build_context(match_uuid)
                        tree_lookup_results = match_context.to_tree_lookup_results_string()
                        relationship_context = match_context.to_relationship_context_string()
                except Exception as exc:  # pragma: no cover
                    logger.debug("Tree context enrichment skipped (non-fatal): %s", exc)

                generated_reply = generate_genealogical_reply(
                    context_history,
                    message_content,
                    genealogical_data_str,
                    self.session_manager,
                    tree_lookup_results=tree_lookup_results,
                    relationship_context=relationship_context,
                )

                if generated_reply:
                    self._persist_outbound_generated_reply_log(
                        person=person,
                        conversation_id=conversation_id,
                        reply_content=generated_reply,
                        intent=intent,
                        research_matches_count=len(research_results),
                        semantic_search_ran=semantic_search is not None,
                    )

        return research_results, generated_reply, extracted_data, semantic_search

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

    @staticmethod
    def _map_field_name_for_conflict(fact_type: str) -> str:
        mapping = {
            "BIRTH": "birth_year",
            "DEATH": "death_year",
            "MARRIAGE": "marriage_date",
            "RELATIONSHIP": "relationship",
            "LOCATION": "location",
        }
        return mapping.get(fact_type, fact_type.lower())

    @staticmethod
    def _resolve_fact_type_enum(fact_type: str) -> FactTypeEnum:
        try:
            return FactTypeEnum(fact_type)
        except Exception:
            return FactTypeEnum.OTHER

    @staticmethod
    def _structured_value_for_fact(fact: ExtractedFact) -> str:
        return fact.normalized_value or fact.structured_value or ""

    def _harvest_facts(
        self,
        person: Person,
        extracted_data: dict[str, Any],
        *,
        original_message: str,
        conversation_id: str,
        source_message_id: Optional[int],
    ) -> None:
        """Validate extracted facts and stage SuggestedFact/DataConflict records."""

        facts: list[ExtractedFact] = extract_facts_from_ai_response(
            extracted_data,
            conversation_id=conversation_id,
            original_message=original_message,
        )

        data = extracted_data.get("extracted_data", {})
        for key_fact in data.get("key_facts", []):
            try:
                inferred = self._infer_fact_type(str(key_fact))
                fact_type = inferred.value
            except Exception:
                fact_type = FactTypeEnum.OTHER.value

            facts.append(
                ExtractedFact(
                    fact_type=fact_type,
                    subject_name=str(getattr(person, "display_name", "") or getattr(person, "id", "")),
                    original_text=original_message,
                    structured_value=str(key_fact),
                    normalized_value=str(key_fact).strip(),
                    confidence=80,
                    source_conversation_id=conversation_id,
                )
            )

        if not facts:
            return

        validator = FactValidator(db_session=self.db)

        for fact in facts:
            try:
                validation_result = validator.validate_fact(fact, person)
            except Exception as exc:  # pragma: no cover
                logger.debug("Fact validation failed (non-fatal): %s", exc)
                continue

            structured_value = self._structured_value_for_fact(fact)
            fact_type_enum = self._resolve_fact_type_enum(fact.fact_type)
            status = FactStatusEnum.APPROVED if validation_result.auto_approved else FactStatusEnum.PENDING

            fact_source_id = str(source_message_id) if source_message_id is not None else conversation_id

            self.db.add(
                SuggestedFact(
                    people_id=person.id,
                    fact_type=fact_type_enum,
                    original_value=fact.original_text,
                    new_value=structured_value,
                    source_message_id=fact_source_id,
                    status=status,
                    confidence_score=fact.confidence,
                )
            )

            if validation_result.conflict_type in {ConflictType.MINOR_CONFLICT, ConflictType.MAJOR_CONFLICT}:
                _record_messaging_counter(
                    "validation_conflicts",
                    labels={
                        "source": "inbound",
                        "conflict_type": str(getattr(validation_result.conflict_type, "value", "UNKNOWN")),
                    },
                )
                existing_value = (
                    validation_result.conflicting_fact.value if validation_result.conflicting_fact else None
                )
                self.db.add(
                    DataConflict(
                        people_id=person.id,
                        field_name=self._map_field_name_for_conflict(fact.fact_type),
                        existing_value=existing_value,
                        new_value=structured_value,
                        source="conversation",
                        source_message_id=source_message_id,
                        confidence_score=fact.confidence,
                        status=ConflictStatusEnum.OPEN,
                    )
                )

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
            current_summary = state.ai_summary or ""

            if safety_result.status == SafetyStatus.OPT_OUT:
                # Hard stop: never contact again.
                state.status = ConversationStatusEnum.OPT_OUT
                state.safety_flag = False
                state.ai_summary = f"OPT_OUT: {safety_result.reason}\n{current_summary}"
                with suppress(Exception):
                    person.status = PersonStatusEnum.DESIST
                    person.automation_enabled = False
            else:
                # Safety/critical/human review.
                state.status = ConversationStatusEnum.HUMAN_REVIEW
                state.safety_flag = True
                state.ai_summary = f"SAFETY FLAG: {safety_result.reason}\n{current_summary}"
                with suppress(Exception):
                    # Use BLOCKED as a safety lock (can be manually cleared later).
                    if getattr(person, "status", None) != PersonStatusEnum.DESIST:
                        person.status = PersonStatusEnum.BLOCKED
                    person.automation_enabled = False

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
    """Run comprehensive unit tests for the InboundOrchestrator."""
    import unittest

    from messaging.test_inbound import TestInboundOrchestrator
    from testing.test_framework import TestSuite

    def _run_unittest_case(case_cls: type[unittest.TestCase], case_name: str) -> None:
        """Execute a unittest.TestCase method and raise on failure for standardized reporting."""
        result = unittest.TestResult()
        case = case_cls(methodName=case_name)
        case.run(result)

        if result.failures or result.errors:
            failure_case, failure_trace = (result.failures + result.errors)[0]
            summary = failure_trace.splitlines()[-1] if failure_trace else "Test case failed"
            raise AssertionError(f"{failure_case.id()} failed: {summary}")

    suite = TestSuite("Inbound Orchestrator", "messaging/inbound.py")
    suite.start_suite()

    loader = unittest.TestLoader()
    for test_name in loader.getTestCaseNames(TestInboundOrchestrator):
        suite.run_test(
            test_name=f"TestInboundOrchestrator.{test_name}",
            test_func=lambda name=test_name: _run_unittest_case(TestInboundOrchestrator, name),
            test_summary=f"Execute unittest method '{test_name}' with standardized reporting",
            expected_outcome="Underlying unittest completes with no failures or errors",
        )

    return suite.finish_suite()


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
