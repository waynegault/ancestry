import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from ai.ai_interface import classify_message_intent
from core.session_manager import SessionManager
from database import ConversationState, ConversationStatusEnum, Person
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

        # 1. Safety Check
        safety_result = self.safety_guard.check_message(message_content)
        if safety_result.status != SafetyStatus.SAFE:
            logger.warning(f"Safety check failed for message from {sender_id}: {safety_result.reason}")
            self._handle_unsafe_message(sender_id, safety_result)
            return {
                "status": "unsafe",
                "safety_result": safety_result,
                "action": "flagged",
            }

        # 2. Intent Classification
        intent = classify_message_intent(context_history, self.session_manager)
        logger.info(f"Classified intent for {sender_id}: {intent}")

        # Update Conversation State
        self._update_conversation_state(sender_id, intent)

        # 3. Genealogical Research (if applicable)
        research_results = None
        if intent in {"PRODUCTIVE", "ENTHUSIASTIC"}:
            # TODO: Extract entities and perform search
            # For now, we just log that we would do research
            logger.info("Intent warrants research. (Research logic to be implemented)")
            pass

        return {
            "status": "processed",
            "intent": intent,
            "safety_result": safety_result,
            "research_results": research_results,
        }

    def _handle_unsafe_message(self, sender_id: str, safety_result: SafetyCheckResult) -> None:
        """Handle a message that failed the safety check."""
        # Update conversation state to flagged
        state = self._get_or_create_conversation_state(sender_id)
        if state:
            state.safety_flag = True
            # Append reason to ai_summary
            current_summary = state.ai_summary or ""
            state.ai_summary = f"SAFETY FLAG: {safety_result.reason}\n{current_summary}"
            state.status = ConversationStatusEnum.HUMAN_REVIEW
            self.db.commit()

    def _update_conversation_state(self, sender_id: str, intent: Optional[str]) -> None:
        """Update the conversation state in the database."""
        state = self._get_or_create_conversation_state(sender_id)
        if state:
            state.last_intent = intent

            # Update status based on intent
            if intent == "DESIST":
                state.status = ConversationStatusEnum.OPT_OUT
            elif intent == "PRODUCTIVE":
                state.status = ConversationStatusEnum.ACTIVE

            self.db.commit()

    def _get_or_create_conversation_state(self, sender_id: str) -> Optional[ConversationState]:
        """Get existing conversation state or create a new one."""
        # Find person by profile_id or uuid
        person = self.db.query(Person).filter((Person.profile_id == sender_id) | (Person.uuid == sender_id)).first()

        if not person:
            logger.warning(f"Could not find Person with ID {sender_id}. Cannot create ConversationState.")
            return None

        state = self.db.query(ConversationState).filter(ConversationState.people_id == person.id).first()
        if not state:
            state = ConversationState(
                people_id=person.id,
                status=ConversationStatusEnum.ACTIVE,
            )
            self.db.add(state)
            self.db.commit()
        return state
