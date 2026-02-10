#!/usr/bin/env python3
"""Action 16: Unified Send - Consolidated outbound messaging.

Processes ALL outbound messages in a single pass with priority ordering:
1. DESIST Acknowledgements - Person opted out, send acknowledgement
2. Approved Drafts - Human-reviewed drafts ready to send
3. AI Replies - Productive conversations needing custom response
4. Template Sequences - Automated Initial/Follow-Up/Final messages

This action replaces the need to run Actions 8, 9, and 11 separately.
It uses the MessageSendOrchestrator for all sends, ensuring consistent
safety checks, database updates, and audit trails.

Usage:
    python main.py  # Select option 16

Features:
    - Single pass through all messaging needs
    - Priority-based processing (one message per person max)
    - Unified safety checks via orchestrator
    - Comprehensive logging and metrics
    - Rate limiting applied automatically
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, not_, or_
from sqlalchemy.orm import Session

from config import config_schema
from core.app_mode_policy import should_allow_outbound_to_person
from core.database import (
    ConversationLog,
    ConversationState,
    DraftReply,
    MessageDirectionEnum,
    Person,
    PersonStatusEnum,
)
from messaging.send_orchestrator import (
    MessageSendContext,
    MessageSendOrchestrator,
    SendResult,
    SendTrigger,
    create_action8_context,
    create_action9_context,
    create_action11_context,
    create_desist_context,
)
from testing.test_framework import TestSuite

if TYPE_CHECKING:
    from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------------------


class MessagePriority(Enum):
    """Priority order for message types. Lower number = higher priority."""

    DESIST_ACK = 1
    APPROVED_DRAFT = 2
    AI_REPLY = 3
    TEMPLATE_SEQUENCE = 4


# ------------------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------------------


@dataclass
class SendCandidate:
    """A person who may need a message, with determined priority."""

    person: Person
    priority: MessagePriority
    context: MessageSendContext
    source_data: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: SendCandidate) -> bool:
        """Sort by priority (lower = higher priority)."""
        return self.priority.value < other.priority.value


@dataclass
class UnifiedSendResult:
    """Result of the unified send operation."""

    total_candidates: int = 0
    sent_count: int = 0
    blocked_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    by_priority: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ------------------------------------------------------------------------------
# UnifiedSendProcessor Class
# ------------------------------------------------------------------------------


class UnifiedSendProcessor:
    """
    Unified processor for all outbound messaging.

    Gathers all people needing messages, determines priority, and processes
    in a single pass using the MessageSendOrchestrator.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize the processor.

        Args:
            session_manager: SessionManager for database and API access.
        """
        self._session_manager = session_manager
        self._orchestrator = MessageSendOrchestrator(session_manager)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def db_session(self) -> Session:
        """Get the current database session."""
        session = self._session_manager.get_db_conn()
        if session is None:
            raise RuntimeError("Database session not available")
        return session

    def process(self, max_sends: int | None = None) -> UnifiedSendResult:
        """
        Process all outbound messaging needs.

        Args:
            max_sends: Optional limit on number of messages to send.

        Returns:
            UnifiedSendResult with counts and any errors.
        """
        start_time = time.time()
        result = UnifiedSendResult()

        self._logger.info("=" * 60)
        self._logger.info("UNIFIED SEND: Starting consolidated outbound processing")
        self._logger.info("=" * 60)

        try:
            # Gather all candidates
            candidates = self._gather_all_candidates()
            result.total_candidates = len(candidates)
            self._logger.info(f"Found {len(candidates)} candidates for messaging")

            if not candidates:
                self._logger.info("No candidates found - nothing to send")
                result.duration_seconds = time.time() - start_time
                return result

            # Sort by priority
            candidates.sort()

            # Process each candidate
            processed_person_ids: set[int] = set()
            send_count = 0

            for candidate in candidates:
                # Skip if we've already processed this person
                if candidate.person.id in processed_person_ids:
                    result.skipped_count += 1
                    continue

                # Check send limit
                if max_sends is not None and send_count >= max_sends:
                    self._logger.info(f"Reached max_sends limit ({max_sends})")
                    break

                # Process this candidate
                priority_name = candidate.priority.name
                success = self._process_candidate(candidate, result)

                if success:
                    result.sent_count += 1
                    result.by_priority[priority_name] = result.by_priority.get(priority_name, 0) + 1
                    send_count += 1

                processed_person_ids.add(candidate.person.id)

        except Exception as e:
            self._logger.error(f"Unified send failed: {e}")
            result.errors.append(str(e))
            result.error_count += 1

        result.duration_seconds = time.time() - start_time
        self._log_summary(result)
        return result

    def _gather_all_candidates(self) -> list[SendCandidate]:
        """Gather all people needing messages across all categories."""
        candidates: list[SendCandidate] = []

        # 1. DESIST Acknowledgements (highest priority)
        desist_candidates = self._gather_desist_candidates()
        candidates.extend(desist_candidates)
        self._logger.debug(f"Found {len(desist_candidates)} DESIST acknowledgements needed")

        # 2. Approved Drafts
        draft_candidates = self._gather_approved_draft_candidates()
        candidates.extend(draft_candidates)
        self._logger.debug(f"Found {len(draft_candidates)} approved drafts to send")

        # 3. AI Replies (productive conversations needing response)
        reply_candidates = self._gather_ai_reply_candidates()
        candidates.extend(reply_candidates)
        self._logger.debug(f"Found {len(reply_candidates)} AI replies needed")

        # 4. Template Sequences (automated messaging)
        sequence_candidates = self._gather_sequence_candidates()
        candidates.extend(sequence_candidates)
        self._logger.debug(f"Found {len(sequence_candidates)} template sequences to process")

        return candidates

    def _gather_desist_candidates(self) -> list[SendCandidate]:
        """Find people with DESIST status who need acknowledgement."""
        candidates: list[SendCandidate] = []

        try:
            # Find DESIST persons without acknowledgement sent
            desist_persons = (
                self.db_session.query(Person)
                .filter(
                    Person.status == PersonStatusEnum.DESIST,
                    Person.desist_acknowledged_at.is_(None),  # No acknowledgement sent yet
                )
                .all()
            )

            for person in desist_persons:
                # Check app mode policy - all sends must respect app_mode
                if not should_allow_outbound_to_person(person, logger):
                    continue

                context = create_desist_context(
                    person=person,
                    conversation_logs=[],
                )
                candidates.append(
                    SendCandidate(
                        person=person,
                        priority=MessagePriority.DESIST_ACK,
                        context=context,
                        source_data={"type": "desist_ack"},
                    )
                )

        except Exception as e:
            self._logger.warning(f"Error gathering DESIST candidates: {e}")

        return candidates

    def _gather_approved_draft_candidates(self) -> list[SendCandidate]:
        """Find approved drafts ready to send."""
        candidates: list[SendCandidate] = []

        try:
            # Find approved drafts (status is a string, not an enum)
            approved_drafts = (
                self.db_session.query(DraftReply)
                .filter(
                    DraftReply.status == "APPROVED",
                )
                .all()
            )

            for draft in approved_drafts:
                # Get the person
                person = self.db_session.query(Person).filter(Person.id == draft.person_id).first()
                if not person:
                    continue

                # Check app mode policy - all sends must respect app_mode
                if not should_allow_outbound_to_person(person, logger):
                    continue

                # Get conversation logs for context
                conv_logs = (
                    self.db_session.query(ConversationLog)
                    .filter(ConversationLog.people_id == person.id)
                    .order_by(ConversationLog.latest_timestamp.desc())
                    .limit(10)
                    .all()
                )

                context = create_action11_context(
                    person=person,
                    conversation_logs=conv_logs,
                    draft_content=draft.draft_text,
                    draft_id=draft.id,
                )
                candidates.append(
                    SendCandidate(
                        person=person,
                        priority=MessagePriority.APPROVED_DRAFT,
                        context=context,
                        source_data={"type": "approved_draft", "draft_id": draft.id},
                    )
                )

        except Exception as e:
            self._logger.warning(f"Error gathering approved draft candidates: {e}")

        return candidates

    def _gather_ai_reply_candidates(self) -> list[SendCandidate]:
        """Find productive conversations needing AI replies."""
        candidates: list[SendCandidate] = []

        try:
            # Find people with recent inbound productive messages needing reply
            # A "productive" conversation is one with ACTIVE status and recent inbound
            lookback = datetime.now(UTC) - timedelta(days=7)

            # Get conversations with recent inbound messages
            recent_inbound = (
                self.db_session.query(ConversationLog)
                .filter(
                    ConversationLog.direction == MessageDirectionEnum.IN,
                    ConversationLog.latest_timestamp >= lookback,
                )
                .order_by(ConversationLog.latest_timestamp.desc())
                .all()
            )

            # Group by person, check if reply needed
            person_last_inbound: dict[int, ConversationLog] = {}
            for log in recent_inbound:
                if log.people_id not in person_last_inbound:
                    person_last_inbound[log.people_id] = log

            for person_id, last_inbound in person_last_inbound.items():
                # Check if we already replied
                reply_after = (
                    self.db_session.query(ConversationLog)
                    .filter(
                        ConversationLog.people_id == person_id,
                        ConversationLog.direction == MessageDirectionEnum.OUT,
                        ConversationLog.latest_timestamp > last_inbound.latest_timestamp,
                    )
                    .first()
                )

                if reply_after:
                    continue  # Already replied

                # Get the person
                person = self.db_session.query(Person).filter(Person.id == person_id).first()
                if not person:
                    continue

                # Skip non-active persons
                if person.status not in {PersonStatusEnum.ACTIVE, PersonStatusEnum.PRODUCTIVE}:
                    continue

                # Check app mode policy - all sends must respect app_mode
                if not should_allow_outbound_to_person(person, logger):
                    continue

                # Get conversation logs
                conv_logs = (
                    self.db_session.query(ConversationLog)
                    .filter(ConversationLog.people_id == person_id)
                    .order_by(ConversationLog.latest_timestamp.desc())
                    .limit(10)
                    .all()
                )

                # Note: AI content will be generated by the orchestrator
                context = create_action9_context(
                    person=person,
                    conversation_logs=conv_logs,
                    ai_generated_content="",  # Will be generated
                    ai_context={
                        "inbound_message": last_inbound.message_text,
                        "requires_generation": True,
                    },
                )
                candidates.append(
                    SendCandidate(
                        person=person,
                        priority=MessagePriority.AI_REPLY,
                        context=context,
                        source_data={
                            "type": "ai_reply",
                            "last_inbound_id": last_inbound.id,
                        },
                    )
                )

        except Exception as e:
            self._logger.warning(f"Error gathering AI reply candidates: {e}")

        return candidates

    def _gather_sequence_candidates(self) -> list[SendCandidate]:
        """Find people eligible for template sequence messages."""
        candidates: list[SendCandidate] = []

        try:
            # Find people eligible for automated messaging
            # This replicates the logic from Action 8's person filtering
            eligible_persons = (
                self.db_session.query(Person)
                .filter(
                    Person.status.in_([PersonStatusEnum.ACTIVE, PersonStatusEnum.NEW]),
                    Person.automation_enabled == True,  # noqa: E712
                )
                .all()
            )

            for person in eligible_persons:
                # Check app mode policy
                if not should_allow_outbound_to_person(person, logger):
                    continue

                # Get conversation logs
                conv_logs = (
                    self.db_session.query(ConversationLog)
                    .filter(ConversationLog.people_id == person.id)
                    .order_by(ConversationLog.latest_timestamp.desc())
                    .limit(10)
                    .all()
                )

                # Get conversation state
                conv_state = (
                    self.db_session.query(ConversationState).filter(ConversationState.person_id == person.id).first()
                )

                context = create_action8_context(
                    person=person,
                    conversation_logs=conv_logs,
                    conversation_state=conv_state,
                )
                candidates.append(
                    SendCandidate(
                        person=person,
                        priority=MessagePriority.TEMPLATE_SEQUENCE,
                        context=context,
                        source_data={"type": "template_sequence"},
                    )
                )

        except Exception as e:
            self._logger.warning(f"Error gathering sequence candidates: {e}")

        return candidates

    def _process_candidate(self, candidate: SendCandidate, result: UnifiedSendResult) -> bool:
        """Process a single send candidate.

        Args:
            candidate: The SendCandidate to process.
            result: The UnifiedSendResult to update.

        Returns:
            True if message was sent successfully.
        """
        person = candidate.person
        priority = candidate.priority.name
        log_prefix = f"[Person #{person.id}][{priority}]"

        try:
            self._logger.debug(f"{log_prefix} Processing...")

            # Send via orchestrator
            send_result = self._orchestrator.send(candidate.context)

            if send_result.success:
                self._logger.info(f"{log_prefix} ✅ Message sent successfully")
                return True
            # Blocked by safety checks or other reason
            self._logger.info(f"{log_prefix} ⏭️ Blocked: {send_result.error}")
            result.blocked_count += 1
            return False

        except Exception as e:
            self._logger.error(f"{log_prefix} ❌ Error: {e}")
            result.error_count += 1
            result.errors.append(f"Person #{person.id}: {e}")
            return False

    def _log_summary(self, result: UnifiedSendResult) -> None:
        """Log the final summary."""
        self._logger.info("=" * 60)
        self._logger.info("UNIFIED SEND: Summary")
        self._logger.info("=" * 60)
        self._logger.info(f"Total candidates: {result.total_candidates}")
        self._logger.info(f"Sent: {result.sent_count}")
        self._logger.info(f"Blocked: {result.blocked_count}")
        self._logger.info(f"Errors: {result.error_count}")
        self._logger.info(f"Skipped (duplicates): {result.skipped_count}")
        self._logger.info(f"Duration: {result.duration_seconds:.1f}s")

        if result.by_priority:
            self._logger.info("By priority:")
            for priority, count in result.by_priority.items():
                self._logger.info(f"  {priority}: {count}")

        if result.errors:
            self._logger.warning("Errors encountered:")
            for error in result.errors[:10]:  # Limit to first 10
                self._logger.warning(f"  {error}")


# ------------------------------------------------------------------------------
# Action Entry Point
# ------------------------------------------------------------------------------


def run_unified_send(session_manager: SessionManager, *args: Any) -> bool:
    """
    Main entry point for Action 16: Unified Send.

    Args:
        session_manager: SessionManager for database and API access.
        *args: Optional arguments (e.g., max_sends limit).

    Returns:
        True if action completed successfully.
    """
    # Parse optional max_sends argument
    max_sends: int | None = None
    if args and args[0]:
        try:
            max_sends = int(args[0])
            logger.info(f"Max sends limit: {max_sends}")
        except (ValueError, TypeError):
            pass

    processor = UnifiedSendProcessor(session_manager)
    result = processor.process(max_sends=max_sends)

    # Return success if we had no errors
    return result.error_count == 0


# ------------------------------------------------------------------------------
# Module Tests
# ------------------------------------------------------------------------------


def _module_tests() -> bool:
    """Run module tests."""
    suite = TestSuite("Unified Send Action", "actions/action16_unified_send.py")
    suite.start_suite()

    # Test 1: MessagePriority ordering
    def test_priority_ordering() -> None:
        assert MessagePriority.DESIST_ACK.value < MessagePriority.APPROVED_DRAFT.value
        assert MessagePriority.APPROVED_DRAFT.value < MessagePriority.AI_REPLY.value
        assert MessagePriority.AI_REPLY.value < MessagePriority.TEMPLATE_SEQUENCE.value

    suite.run_test("MessagePriority ordering is correct", test_priority_ordering)

    # Test 2: SendCandidate sorting
    def test_candidate_sorting() -> None:
        from unittest.mock import MagicMock

        mock_person = MagicMock()
        mock_person.id = 1
        mock_context = MagicMock()

        c1 = SendCandidate(person=mock_person, priority=MessagePriority.TEMPLATE_SEQUENCE, context=mock_context)
        c2 = SendCandidate(person=mock_person, priority=MessagePriority.DESIST_ACK, context=mock_context)
        c3 = SendCandidate(person=mock_person, priority=MessagePriority.APPROVED_DRAFT, context=mock_context)

        sorted_candidates = sorted([c1, c2, c3])
        assert sorted_candidates[0].priority == MessagePriority.DESIST_ACK
        assert sorted_candidates[1].priority == MessagePriority.APPROVED_DRAFT
        assert sorted_candidates[2].priority == MessagePriority.TEMPLATE_SEQUENCE

    suite.run_test("SendCandidate sorts by priority", test_candidate_sorting)

    # Test 3: UnifiedSendResult defaults
    def test_result_defaults() -> None:
        result = UnifiedSendResult()
        assert result.total_candidates == 0
        assert result.sent_count == 0
        assert result.blocked_count == 0
        assert result.error_count == 0
        assert result.by_priority == {}

    suite.run_test("UnifiedSendResult has correct defaults", test_result_defaults)

    # Test 4: Process with no candidates returns clean result
    def test_process_empty_candidates() -> None:
        from unittest.mock import MagicMock, patch

        mock_sm = MagicMock()
        mock_session = MagicMock()
        mock_sm.db_manager.get_session.return_value = mock_session
        # Mock empty queries for all candidate sources
        mock_session.query.return_value.filter.return_value.all.return_value = []
        mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = []

        processor = UnifiedSendProcessor(mock_sm)
        with patch.object(processor, '_gather_all_candidates', return_value=[]):
            result = processor.process()
            assert result.total_candidates == 0, "Empty candidates should yield zero total"
            assert result.sent_count == 0, "No sends expected"
            assert result.error_count == 0, "No errors expected"

    suite.run_test("Process with no candidates", test_process_empty_candidates)

    return suite.finish_suite()


# Standard test runner
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(_module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
