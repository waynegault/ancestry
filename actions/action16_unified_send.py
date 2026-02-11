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

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

# Shadow mode log path
ACTION16_SHADOW_LOG_PATH = Path("Logs/action16_shadow_decisions.jsonl")


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
    shadow_logged_count: int = 0
    by_priority: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    shadow_mode: bool = False


@dataclass
class ShadowDecisionLog:
    """A single shadow-mode decision entry for comparison-ready output."""

    timestamp: str
    person_id: int
    person_name: str
    priority: str
    priority_value: int
    source_type: str
    would_send: bool
    block_reason: str | None = None
    message_preview: str | None = None
    source_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


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
        self._shadow_mode = self._check_shadow_mode()

    @staticmethod
    def _check_shadow_mode() -> bool:
        """Check if Action 16 shadow mode is enabled via config."""
        try:
            return bool(getattr(config_schema, "action16_shadow_mode", False))
        except Exception:
            return False

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

        In shadow mode, all send decisions are logged but NOT executed.

        Args:
            max_sends: Optional limit on number of messages to send.

        Returns:
            UnifiedSendResult with counts and any errors.
        """
        start_time = time.time()
        result = UnifiedSendResult()
        result.shadow_mode = self._shadow_mode

        mode_label = "SHADOW MODE (log-only)" if self._shadow_mode else "LIVE"
        self._logger.info("=" * 60)
        self._logger.info(f"UNIFIED SEND [{mode_label}]: Starting consolidated outbound processing")
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

                # Shadow mode: log decision without sending
                if self._shadow_mode:
                    self._log_shadow_decision(candidate, result)
                    processed_person_ids.add(candidate.person.id)
                    continue

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

    def _log_shadow_decision(self, candidate: SendCandidate, result: UnifiedSendResult) -> None:
        """Log a shadow-mode decision without executing the send.

        Records what WOULD have been sent, to whom, and at what priority,
        producing a comparison-ready JSONL log file.
        """
        person = candidate.person
        priority_name = candidate.priority.name
        log_prefix = f"[SHADOW][Person #{person.id}][{priority_name}]"

        # Determine the message preview from context
        msg_preview: str | None = None
        try:
            ctx = candidate.context
            if hasattr(ctx, "message_content") and ctx.message_content:
                msg_preview = str(ctx.message_content)[:200]
            elif hasattr(ctx, "draft_content") and ctx.draft_content:
                msg_preview = str(ctx.draft_content)[:200]
        except Exception:
            pass

        person_name = getattr(person, "display_name", None) or getattr(person, "name", "") or ""

        entry = ShadowDecisionLog(
            timestamp=datetime.now(UTC).isoformat(),
            person_id=person.id,
            person_name=str(person_name),
            priority=priority_name,
            priority_value=candidate.priority.value,
            source_type=candidate.source_data.get("type", "unknown"),
            would_send=True,
            block_reason=None,
            message_preview=msg_preview,
            source_data=candidate.source_data,
        )

        # Write to JSONL log
        try:
            ACTION16_SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ACTION16_SHADOW_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as e:
            self._logger.error(f"{log_prefix} Failed to write shadow log: {e}")

        self._logger.info(
            f"{log_prefix} 👻 WOULD SEND: {entry.source_type} "
            f"to '{person_name}' (priority={candidate.priority.value})"
        )

        result.shadow_logged_count += 1
        result.by_priority[priority_name] = result.by_priority.get(priority_name, 0) + 1

    def _log_summary(self, result: UnifiedSendResult) -> None:
        """Log the final summary."""
        mode_label = "SHADOW MODE" if result.shadow_mode else "LIVE"
        self._logger.info("=" * 60)
        self._logger.info(f"UNIFIED SEND [{mode_label}]: Summary")
        self._logger.info("=" * 60)
        self._logger.info(f"Total candidates: {result.total_candidates}")

        if result.shadow_mode:
            self._logger.info(f"Shadow-logged (would send): {result.shadow_logged_count}")
            self._logger.info(f"Shadow log: {ACTION16_SHADOW_LOG_PATH}")
        else:
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

    # --------------------------------------------------------------------------
    # Integration Tests: Priority Routing & One-Message-Per-Person
    # --------------------------------------------------------------------------

    # Test 5: DESIST_ACK beats AI_REPLY for the same person
    def test_desist_beats_ai_reply() -> None:
        from unittest.mock import MagicMock, patch

        mock_person = MagicMock()
        mock_person.id = 100

        desist_ctx = MagicMock()
        ai_ctx = MagicMock()

        candidates = [
            SendCandidate(person=mock_person, priority=MessagePriority.AI_REPLY, context=ai_ctx),
            SendCandidate(person=mock_person, priority=MessagePriority.DESIST_ACK, context=desist_ctx),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        mock_send_result = MagicMock()
        mock_send_result.success = True

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send", return_value=mock_send_result) as mock_send,
        ):
            result = processor.process()

        # Only one message should be sent (DESIST_ACK, not AI_REPLY)
        assert result.sent_count == 1, f"Expected 1 send, got {result.sent_count}"
        assert result.skipped_count == 1, f"Expected 1 skipped, got {result.skipped_count}"
        # The context passed to send() should be the DESIST one (sorted first)
        mock_send.assert_called_once_with(desist_ctx)

    suite.run_test(
        "DESIST_ACK takes priority over AI_REPLY for same person",
        test_desist_beats_ai_reply,
    )

    # Test 6: APPROVED_DRAFT beats TEMPLATE_SEQUENCE for the same person
    def test_approved_draft_beats_template() -> None:
        from unittest.mock import MagicMock, patch

        mock_person = MagicMock()
        mock_person.id = 200

        draft_ctx = MagicMock()
        template_ctx = MagicMock()

        candidates = [
            SendCandidate(person=mock_person, priority=MessagePriority.TEMPLATE_SEQUENCE, context=template_ctx),
            SendCandidate(person=mock_person, priority=MessagePriority.APPROVED_DRAFT, context=draft_ctx),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        mock_send_result = MagicMock()
        mock_send_result.success = True

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send", return_value=mock_send_result) as mock_send,
        ):
            result = processor.process()

        assert result.sent_count == 1, f"Expected 1 send, got {result.sent_count}"
        assert result.skipped_count == 1, f"Expected 1 skipped, got {result.skipped_count}"
        mock_send.assert_called_once_with(draft_ctx)

    suite.run_test(
        "APPROVED_DRAFT takes priority over TEMPLATE_SEQUENCE for same person",
        test_approved_draft_beats_template,
    )

    # Test 7: One-message-per-person with all four priority types
    def test_one_message_per_person_all_priorities() -> None:
        from unittest.mock import MagicMock, patch

        mock_person = MagicMock()
        mock_person.id = 300

        contexts = {p: MagicMock() for p in MessagePriority}

        candidates = [
            SendCandidate(person=mock_person, priority=p, context=contexts[p])
            for p in MessagePriority
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        mock_send_result = MagicMock()
        mock_send_result.success = True

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send", return_value=mock_send_result) as mock_send,
        ):
            result = processor.process()

        assert result.sent_count == 1, f"Expected exactly 1 send, got {result.sent_count}"
        assert result.skipped_count == 3, f"Expected 3 skipped, got {result.skipped_count}"
        # Highest priority (DESIST_ACK) should win
        mock_send.assert_called_once_with(contexts[MessagePriority.DESIST_ACK])

    suite.run_test(
        "One-message-per-person with all 4 priority types",
        test_one_message_per_person_all_priorities,
    )

    # Test 8: Multiple different persons each get exactly one message
    def test_multiple_persons_each_get_one() -> None:
        from unittest.mock import MagicMock, patch

        person_a = MagicMock()
        person_a.id = 401
        person_b = MagicMock()
        person_b.id = 402

        ctx_a_template = MagicMock()
        ctx_a_ai = MagicMock()
        ctx_b_desist = MagicMock()
        ctx_b_draft = MagicMock()

        candidates = [
            SendCandidate(person=person_a, priority=MessagePriority.TEMPLATE_SEQUENCE, context=ctx_a_template),
            SendCandidate(person=person_a, priority=MessagePriority.AI_REPLY, context=ctx_a_ai),
            SendCandidate(person=person_b, priority=MessagePriority.APPROVED_DRAFT, context=ctx_b_draft),
            SendCandidate(person=person_b, priority=MessagePriority.DESIST_ACK, context=ctx_b_desist),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        mock_send_result = MagicMock()
        mock_send_result.success = True

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send", return_value=mock_send_result) as mock_send,
        ):
            result = processor.process()

        assert result.sent_count == 2, f"Expected 2 sends, got {result.sent_count}"
        assert result.skipped_count == 2, f"Expected 2 skipped, got {result.skipped_count}"
        # Both persons should be in by_priority
        sent_contexts = [call.args[0] for call in mock_send.call_args_list]
        # Person B's DESIST_ACK (priority 1) processed first, then Person A's AI_REPLY (priority 3)
        assert ctx_b_desist in sent_contexts, "Person B should get DESIST_ACK"
        assert ctx_a_ai in sent_contexts, "Person A should get AI_REPLY"
        assert ctx_a_template not in sent_contexts, "Person A's TEMPLATE should be skipped"
        assert ctx_b_draft not in sent_contexts, "Person B's DRAFT should be skipped"

    suite.run_test(
        "Multiple persons each receive exactly one message",
        test_multiple_persons_each_get_one,
    )

    # Test 9: max_sends limits total sends across persons
    def test_max_sends_limit() -> None:
        from unittest.mock import MagicMock, patch

        persons = []
        candidates = []
        for i in range(5):
            p = MagicMock()
            p.id = 500 + i
            persons.append(p)
            ctx = MagicMock()
            candidates.append(
                SendCandidate(person=p, priority=MessagePriority.TEMPLATE_SEQUENCE, context=ctx)
            )

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        mock_send_result = MagicMock()
        mock_send_result.success = True

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send", return_value=mock_send_result) as mock_send,
        ):
            result = processor.process(max_sends=2)

        assert result.sent_count == 2, f"Expected 2 sends with max_sends=2, got {result.sent_count}"
        assert mock_send.call_count == 2, f"Expected 2 send calls, got {mock_send.call_count}"

    suite.run_test("max_sends limits total sends", test_max_sends_limit)

    # Test 10: Blocked send does not count toward max_sends
    def test_blocked_not_counted_as_sent() -> None:
        from unittest.mock import MagicMock, patch

        person_blocked = MagicMock()
        person_blocked.id = 601
        person_ok = MagicMock()
        person_ok.id = 602

        ctx_blocked = MagicMock()
        ctx_ok = MagicMock()

        candidates = [
            SendCandidate(person=person_blocked, priority=MessagePriority.DESIST_ACK, context=ctx_blocked),
            SendCandidate(person=person_ok, priority=MessagePriority.DESIST_ACK, context=ctx_ok),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        blocked_result = MagicMock()
        blocked_result.success = False
        blocked_result.error = "Safety check failed"
        success_result = MagicMock()
        success_result.success = True

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(
                processor._orchestrator,
                "send",
                side_effect=[blocked_result, success_result],
            ),
        ):
            result = processor.process()

        assert result.sent_count == 1, f"Expected 1 sent, got {result.sent_count}"
        assert result.blocked_count == 1, f"Expected 1 blocked, got {result.blocked_count}"

    suite.run_test(
        "Blocked sends count as blocked not sent",
        test_blocked_not_counted_as_sent,
    )

    # Test 11: by_priority dict tracks correct categories
    def test_by_priority_tracking() -> None:
        from unittest.mock import MagicMock, patch

        p1 = MagicMock()
        p1.id = 701
        p2 = MagicMock()
        p2.id = 702
        p3 = MagicMock()
        p3.id = 703

        candidates = [
            SendCandidate(person=p1, priority=MessagePriority.DESIST_ACK, context=MagicMock()),
            SendCandidate(person=p2, priority=MessagePriority.DESIST_ACK, context=MagicMock()),
            SendCandidate(person=p3, priority=MessagePriority.APPROVED_DRAFT, context=MagicMock()),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        mock_send_result = MagicMock()
        mock_send_result.success = True

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send", return_value=mock_send_result),
        ):
            result = processor.process()

        assert result.by_priority.get("DESIST_ACK") == 2, (
            f"Expected 2 DESIST_ACK, got {result.by_priority.get('DESIST_ACK')}"
        )
        assert result.by_priority.get("APPROVED_DRAFT") == 1, (
            f"Expected 1 APPROVED_DRAFT, got {result.by_priority.get('APPROVED_DRAFT')}"
        )

    suite.run_test("by_priority tracks send categories correctly", test_by_priority_tracking)

    # Test 12: Error during send is recorded without crashing
    def test_send_error_recorded() -> None:
        from unittest.mock import MagicMock, patch

        mock_person = MagicMock()
        mock_person.id = 800

        candidates = [
            SendCandidate(person=mock_person, priority=MessagePriority.AI_REPLY, context=MagicMock()),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(
                processor._orchestrator,
                "send",
                side_effect=RuntimeError("Network failure"),
            ),
        ):
            result = processor.process()

        assert result.error_count == 1, f"Expected 1 error, got {result.error_count}"
        assert result.sent_count == 0, f"Expected 0 sent, got {result.sent_count}"
        assert len(result.errors) == 1, f"Expected 1 error message, got {len(result.errors)}"
        assert "800" in result.errors[0], "Error should reference person ID"

    suite.run_test("Send error is recorded without crashing", test_send_error_recorded)

    # Test 13: Person with no valid candidates (all blocked by orchestrator)
    def test_all_candidates_blocked() -> None:
        from unittest.mock import MagicMock, patch

        p1 = MagicMock()
        p1.id = 901
        p2 = MagicMock()
        p2.id = 902

        candidates = [
            SendCandidate(person=p1, priority=MessagePriority.DESIST_ACK, context=MagicMock()),
            SendCandidate(person=p2, priority=MessagePriority.TEMPLATE_SEQUENCE, context=MagicMock()),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        blocked = MagicMock()
        blocked.success = False
        blocked.error = "Daily limit"

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send", return_value=blocked),
        ):
            result = processor.process()

        assert result.sent_count == 0, f"Expected 0 sent, got {result.sent_count}"
        assert result.blocked_count == 2, f"Expected 2 blocked, got {result.blocked_count}"
        assert result.total_candidates == 2

    suite.run_test(
        "All candidates blocked results in zero sends",
        test_all_candidates_blocked,
    )

    # Test 14: Duration is recorded
    def test_duration_recorded() -> None:
        from unittest.mock import MagicMock, patch

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)

        with patch.object(processor, "_gather_all_candidates", return_value=[]):
            result = processor.process()

        assert result.duration_seconds >= 0, "Duration should be non-negative"

    suite.run_test("Duration is recorded in result", test_duration_recorded)

    # --------------------------------------------------------------------------
    # Shadow Mode Tests
    # --------------------------------------------------------------------------

    # Test 15: Shadow mode prevents actual sends
    def test_shadow_mode_no_sends() -> None:
        from unittest.mock import MagicMock, patch

        mock_person = MagicMock()
        mock_person.id = 1001
        mock_person.display_name = "Shadow Test Person"

        candidates = [
            SendCandidate(
                person=mock_person,
                priority=MessagePriority.DESIST_ACK,
                context=MagicMock(),
                source_data={"type": "desist_ack"},
            ),
            SendCandidate(
                person=mock_person,
                priority=MessagePriority.AI_REPLY,
                context=MagicMock(),
                source_data={"type": "ai_reply"},
            ),
        ]

        mock_sm = MagicMock()
        processor = UnifiedSendProcessor(mock_sm)
        processor._shadow_mode = True  # Force shadow mode

        with (
            patch.object(processor, "_gather_all_candidates", return_value=candidates),
            patch.object(processor._orchestrator, "send") as mock_send,
            patch("builtins.open", MagicMock()),
        ):
            result = processor.process()

        # Orchestrator.send should NEVER be called in shadow mode
        mock_send.assert_not_called()
        assert result.shadow_mode is True, "Result should indicate shadow mode"
        assert result.sent_count == 0, "No actual sends in shadow mode"
        assert result.shadow_logged_count == 1, f"Expected 1 shadow log (dedup), got {result.shadow_logged_count}"

    suite.run_test("Shadow mode prevents actual sends", test_shadow_mode_no_sends)

    # Test 16: Shadow mode result defaults
    def test_shadow_result_defaults() -> None:
        result = UnifiedSendResult()
        assert result.shadow_mode is False, "Shadow mode off by default"
        assert result.shadow_logged_count == 0, "Shadow count starts at 0"

    suite.run_test("UnifiedSendResult shadow defaults", test_shadow_result_defaults)

    # Test 17: ShadowDecisionLog serialization
    def test_shadow_decision_log_serialization() -> None:
        entry = ShadowDecisionLog(
            timestamp="2026-02-10T00:00:00+00:00",
            person_id=42,
            person_name="Test Person",
            priority="DESIST_ACK",
            priority_value=1,
            source_type="desist_ack",
            would_send=True,
        )
        d = entry.to_dict()
        assert d["person_id"] == 42
        assert d["priority"] == "DESIST_ACK"
        assert d["would_send"] is True
        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert "42" in serialized

    suite.run_test("ShadowDecisionLog serializes to JSON", test_shadow_decision_log_serialization)

    # Test 18: Shadow mode config check defaults to False
    def test_shadow_mode_config_default() -> None:
        # _check_shadow_mode uses getattr with default=False
        # Without setting config, it should return False
        result = UnifiedSendProcessor._check_shadow_mode()
        assert result is False, "Shadow mode should be OFF by default"

    suite.run_test("Shadow mode config defaults to False", test_shadow_mode_config_default)

    return suite.finish_suite()


# Standard test runner
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(_module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
