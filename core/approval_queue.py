#!/usr/bin/env python3

"""
Human-in-the-Loop Approval Queue System

Sprint 4: Provides review queue for AI-generated messages before sending.
Implements tiered approval workflow with auto-approval for high-confidence drafts.

Key Features:
- Queue management for draft replies
- Priority-based review ordering
- Auto-approval for high-confidence messages
- Emergency stop controls
- CLI integration for review workflow
"""

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import and_, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)


# === ENUMS ===


class ApprovalStatus(enum.Enum):
    """Status of a draft in the approval queue."""

    PENDING = "PENDING"  # Awaiting human review
    APPROVED = "APPROVED"  # Approved for sending
    REJECTED = "REJECTED"  # Rejected, will not send
    AUTO_APPROVED = "AUTO_APPROVED"  # System auto-approved
    EXPIRED = "EXPIRED"  # Timed out without action
    SENT = "SENT"  # Message has been sent


class ReviewPriority(enum.Enum):
    """Priority levels for review queue ordering."""

    LOW = "low"  # High confidence, routine message
    NORMAL = "normal"  # Standard confidence
    HIGH = "high"  # Low confidence or sensitive
    CRITICAL = "critical"  # Requires immediate attention


# === DATA CLASSES ===


@dataclass
class QueuedDraft:
    """Represents a draft in the review queue."""

    draft_id: int
    person_id: int
    person_name: str
    conversation_id: str
    content: str
    ai_confidence: int
    priority: ReviewPriority
    status: ApprovalStatus
    created_at: datetime
    ai_reasoning: Optional[str] = None
    context_summary: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass
class ReviewDecision:
    """Result of a review action."""

    success: bool
    draft_id: int
    action: str  # "approve", "reject", "edit"
    message: str
    edited_content: Optional[str] = None


@dataclass()
class QueueStats:
    """Statistics about the approval queue."""

    pending_count: int = 0
    auto_approved_count: int = 0
    approved_today: int = 0
    rejected_today: int = 0
    expired_count: int = 0
    by_priority: dict[str, int] = field(default_factory=dict)


# === APPROVAL QUEUE SERVICE ===


class ApprovalQueueService:
    """
    Service for managing the human-in-the-loop approval queue.

    Provides methods for:
    - Queueing draft messages for review
    - Auto-approving high-confidence drafts
    - Manual review workflow
    - Queue statistics and monitoring
    """

    # Confidence thresholds for auto-approval
    AUTO_APPROVE_THRESHOLD = 90
    HIGH_PRIORITY_THRESHOLD = 70

    def __init__(self, db_session: DbSession, auto_approve_enabled: Optional[bool] = None) -> None:
        """Initialize the approval queue service."""
        self.db_session = db_session
        self._auto_approve_enabled = self._resolve_auto_approve_flag(auto_approve_enabled)
        if not self._auto_approve_enabled:
            logger.info("Auto-approval disabled (auto_approve_enabled=False)")

    @staticmethod
    def _resolve_auto_approve_flag(override: Optional[bool]) -> bool:
        """Resolve auto-approve toggle from override or configuration."""
        if override is not None:
            return override

        try:
            from config import config_schema

            return getattr(config_schema, "auto_approve_enabled", False)
        except Exception:
            # Default to safest option if configuration not available
            return False

    @staticmethod
    def _get_owner_profile_id() -> Optional[str]:
        """Get the tree owner's profile ID for self-message prevention."""
        import os

        # First try environment variable
        owner_id = os.getenv("MY_PROFILE_ID")
        if owner_id:
            return owner_id

        # Try SessionManager if available
        try:
            from core.session_manager import SessionManager

            sm = SessionManager()
            if hasattr(sm, "my_profile_id") and sm.my_profile_id:
                return sm.my_profile_id
            if hasattr(sm, "api_manager") and hasattr(sm.api_manager, "my_profile_id"):
                return sm.api_manager.my_profile_id
        except Exception:
            pass

        # Try config schema
        try:
            from config import config_schema

            if hasattr(config_schema, "test") and hasattr(config_schema.test, "test_profile_id"):
                return config_schema.test.test_profile_id
        except Exception:
            pass

        return None

    def queue_for_review(
        self,
        person_id: int,
        conversation_id: str,
        content: str,
        ai_confidence: int,
        _ai_reasoning: Optional[str] = None,
        _context_summary: Optional[str] = None,
        _research_suggestions: Optional[str] = None,
        _research_metadata: Optional[dict[str, Any]] = None,
        expiry_hours: int = 72,
    ) -> Optional[int]:
        """
        Queue a draft message for human review.

        Args:
            person_id: ID of the person the message is for
            conversation_id: Ancestry conversation ID
            content: Draft message content
            ai_confidence: AI confidence score (0-100)
            ai_reasoning: Optional AI explanation for the draft
            context_summary: Optional context used to generate draft
            expiry_hours: Hours until draft expires if not reviewed

        Returns:
            Draft ID if queued, None if auto-approved or failed
        """
        try:
            from core.database import DraftReply, Person
            from core.draft_content import DraftInternalMetadata, append_internal_metadata

            # Check if person exists and is contactable
            person = self.db_session.query(Person).filter(Person.id == person_id).first()
            if not person:
                logger.warning(f"Cannot queue draft: Person {person_id} not found")
                return None

            # Check if person opted out (DESIST status)
            if hasattr(person, "status") and person.status.value == "DESIST":
                logger.warning(f"Cannot queue draft: Person {person_id} has DESIST status")
                return None

            # CRITICAL: Self-message prevention - don't draft messages to ourselves
            owner_profile_id = self._get_owner_profile_id()
            if owner_profile_id and hasattr(person, "profile_id") and person.profile_id:
                if str(person.profile_id) == str(owner_profile_id):
                    logger.error(
                        f"🚫 BLOCKED: Self-message attempt! Person {person_id} "
                        f"(profile_id={person.profile_id}) is the tree owner. "
                        "Draft NOT queued."
                    )
                    return None

            # Embed review-only metadata (no schema migrations).
            content_with_metadata = append_internal_metadata(
                content,
                DraftInternalMetadata(
                    ai_confidence=ai_confidence,
                    ai_reasoning=_ai_reasoning,
                    context_summary=_context_summary,
                    research_suggestions=_research_suggestions,
                    research_metadata=_research_metadata,
                ),
            )

            # Determine priority based on confidence
            priority = self._calculate_priority(ai_confidence, person)

            # De-duplicate: if a pending draft already exists for this conversation/person,
            # update it in-place instead of inserting a new row.
            existing_pending = (
                self.db_session.query(DraftReply)
                .filter(
                    DraftReply.people_id == person_id,
                    DraftReply.conversation_id == conversation_id,
                    DraftReply.status == "PENDING",
                )
                .first()
            )

            if existing_pending is not None:
                if existing_pending.content != content_with_metadata:
                    existing_pending.content = content_with_metadata
                    self.db_session.add(existing_pending)
                    self.db_session.commit()
                    logger.info(
                        "Updated existing pending draft %s for review (confidence=%s, priority=%s)",
                        existing_pending.id,
                        ai_confidence,
                        priority.value,
                    )
                else:
                    logger.debug(
                        "Skipped updating pending draft %s (content unchanged)",
                        existing_pending.id,
                    )
                return existing_pending.id

            # Check for auto-approval
            if self._should_auto_approve(ai_confidence, priority, person):
                return self._create_auto_approved_draft(
                    person_id,
                    conversation_id,
                    content_with_metadata,
                    ai_confidence,
                )

            # Calculate expiry time for PENDING drafts
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

            draft = DraftReply(
                people_id=person_id,
                conversation_id=conversation_id,
                content=content_with_metadata,
                status="PENDING",
                expires_at=expires_at,
            )
            self.db_session.add(draft)
            self.db_session.commit()

            logger.info(f"Queued draft {draft.id} for review (confidence={ai_confidence}, priority={priority.value})")
            return draft.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to queue draft for review: {e}")
            self.db_session.rollback()
            return None

    def _calculate_priority(self, confidence: int, person: Any) -> ReviewPriority:
        """Calculate review priority based on confidence and person context."""
        # Critical: Previous DESIST or sensitive history
        if hasattr(person, "status") and person.status.value in {"DESIST", "BLOCKED"}:
            return ReviewPriority.CRITICAL

        # High: Low confidence
        if confidence < self.HIGH_PRIORITY_THRESHOLD:
            return ReviewPriority.HIGH

        # Low: Very high confidence
        if confidence >= self.AUTO_APPROVE_THRESHOLD:
            return ReviewPriority.LOW

        return ReviewPriority.NORMAL

    def _should_auto_approve(self, confidence: int, priority: ReviewPriority, person: Any) -> bool:
        """Determine if draft should be auto-approved."""
        if not self._auto_approve_enabled:
            return False

        if priority in {ReviewPriority.CRITICAL, ReviewPriority.HIGH}:
            return False

        if confidence < self.AUTO_APPROVE_THRESHOLD:
            return False

        # First message to a person should never auto-approve
        return not self._is_first_message(person)

    def _is_first_message(self, person: Any) -> bool:
        """Check if this would be the first message to a person."""
        try:
            from core.database import ConversationLog, MessageDirectionEnum

            outbound_count = (
                self.db_session.query(func.count(ConversationLog.id))
                .filter(
                    and_(
                        ConversationLog.people_id == person.id,
                        ConversationLog.direction == MessageDirectionEnum.OUT,
                    )
                )
                .scalar()
            )
            return outbound_count == 0
        except Exception:
            # If we can't determine, be safe
            return True

    def _create_auto_approved_draft(
        self, person_id: int, conversation_id: str, content: str, confidence: int
    ) -> Optional[int]:
        """Create an auto-approved draft."""
        try:
            from core.database import DraftReply

            draft = DraftReply(
                people_id=person_id,
                conversation_id=conversation_id,
                content=content,
                status="AUTO_APPROVED",
            )
            self.db_session.add(draft)
            self.db_session.commit()

            logger.info(f"Auto-approved draft {draft.id} (confidence={confidence})")
            return draft.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to create auto-approved draft: {e}")
            self.db_session.rollback()
            return None

    def get_pending_queue(self, limit: int = 50, _priority_filter: Optional[str] = None) -> list[QueuedDraft]:
        """
        Get pending drafts for review, ordered by priority and age.

        Args:
            limit: Maximum number of drafts to return
            _priority_filter: Optional priority level to filter by (reserved for future use)

        Returns:
            List of QueuedDraft objects
        """
        try:
            from core.database import DraftReply, Person

            query = (
                self.db_session.query(DraftReply, Person)
                .join(Person, DraftReply.people_id == Person.id)
                .filter(DraftReply.status == "PENDING")
            )

            # Apply priority filter if specified
            # Note: We'd need to add priority column to DraftReply for full implementation

            query = query.order_by(DraftReply.created_at.asc())
            query = query.limit(limit)

            results: list[QueuedDraft] = []
            for draft, person in query.all():
                queued = QueuedDraft(
                    draft_id=draft.id,
                    person_id=person.id,
                    person_name=person.display_name,
                    conversation_id=draft.conversation_id,
                    content=draft.content,
                    ai_confidence=80,  # Default; would come from draft if stored
                    priority=ReviewPriority.NORMAL,
                    status=self._parse_draft_status(draft.status),
                    created_at=draft.created_at,
                )
                results.append(queued)

            return results

        except SQLAlchemyError as e:
            logger.error(f"Failed to get pending queue: {e}")
            return []

    @staticmethod
    def _parse_draft_status(status: str) -> ApprovalStatus:
        """Parse DraftReply.status into ApprovalStatus with backward compatibility."""
        if status == "DISCARDED":
            return ApprovalStatus.REJECTED
        try:
            return ApprovalStatus(status)
        except Exception:
            logger.debug("Unknown DraftReply status '%s'; defaulting to PENDING", status)
            return ApprovalStatus.PENDING

    def approve(
        self, draft_id: int, reviewer: str = "operator", edited_content: Optional[str] = None
    ) -> ReviewDecision:
        """
        Approve a draft for sending.

        Args:
            draft_id: ID of the draft to approve
            reviewer: Username of the reviewer
            edited_content: Optional edited content to replace the original

        Returns:
            ReviewDecision with result
        """
        try:
            from core.database import DraftReply

            draft = self.db_session.query(DraftReply).filter(DraftReply.id == draft_id).first()
            if not draft:
                return ReviewDecision(
                    success=False,
                    draft_id=draft_id,
                    action="approve",
                    message=f"Draft {draft_id} not found",
                )

            if draft.status != "PENDING":
                return ReviewDecision(
                    success=False,
                    draft_id=draft_id,
                    action="approve",
                    message=f"Draft {draft_id} is not pending (status: {draft.status})",
                )

            # Update draft
            if edited_content:
                draft.content = edited_content

            draft.status = "APPROVED"
            self.db_session.commit()

            logger.info(f"Draft {draft_id} approved by {reviewer}")
            return ReviewDecision(
                success=True,
                draft_id=draft_id,
                action="approve",
                message=f"Draft {draft_id} approved",
                edited_content=edited_content,
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to approve draft {draft_id}: {e}")
            self.db_session.rollback()
            return ReviewDecision(
                success=False,
                draft_id=draft_id,
                action="approve",
                message=f"Database error: {e}",
            )

    def reject(self, draft_id: int, reviewer: str = "operator", reason: str = "") -> ReviewDecision:
        """
        Reject a draft.

        Args:
            draft_id: ID of the draft to reject
            reviewer: Username of the reviewer
            reason: Reason for rejection

        Returns:
            ReviewDecision with result
        """
        try:
            from core.database import ConversationState, DraftReply
            from observability.conversation_analytics import record_engagement_event

            draft = self.db_session.query(DraftReply).filter(DraftReply.id == draft_id).first()
            if not draft:
                return ReviewDecision(
                    success=False,
                    draft_id=draft_id,
                    action="reject",
                    message=f"Draft {draft_id} not found",
                )

            if draft.status != "PENDING":
                return ReviewDecision(
                    success=False,
                    draft_id=draft_id,
                    action="reject",
                    message=f"Draft {draft_id} is not pending (status: {draft.status})",
                )

            draft.status = "REJECTED"

            conversation_phase: Optional[str] = None
            conv_state = (
                self.db_session.query(ConversationState).filter(ConversationState.people_id == draft.people_id).first()
            )
            if conv_state is not None:
                conversation_phase = conv_state.conversation_phase

                # Avoid mutating hard-stop states.
                status_val = getattr(conv_state, "status", None)
                status_str = str(getattr(status_val, "value", status_val) or "")
                if status_str not in {"OPT_OUT", "HUMAN_REVIEW", "PAUSED"}:
                    conv_state.next_action = "no_action"
                    conv_state.next_action_date = None
                    self.db_session.add(conv_state)

            self.db_session.commit()

            # Record as an engagement event for observability/metrics.
            record_engagement_event(
                session=self.db_session,
                people_id=draft.people_id,
                event_type="draft_rejected",
                event_description="Draft rejected by operator",
                event_data={
                    "draft_id": draft.id,
                    "conversation_id": draft.conversation_id,
                    "reviewer": reviewer,
                    "reason": reason,
                },
                conversation_phase=conversation_phase,
            )

            logger.info(f"Draft {draft_id} rejected by {reviewer}: {reason}")
            return ReviewDecision(
                success=True,
                draft_id=draft_id,
                action="reject",
                message=f"Draft {draft_id} rejected: {reason}",
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to reject draft {draft_id}: {e}")
            self.db_session.rollback()
            return ReviewDecision(
                success=False,
                draft_id=draft_id,
                action="reject",
                message=f"Database error: {e}",
            )

    def get_queue_stats(self) -> QueueStats:
        """Get statistics about the approval queue."""
        try:
            from core.database import DraftReply

            stats = QueueStats()

            # Count by status
            status_counts = (
                self.db_session.query(DraftReply.status, func.count(DraftReply.id)).group_by(DraftReply.status).all()
            )

            for status, count in status_counts:
                if status == "PENDING":
                    stats.pending_count = count
                elif status == "AUTO_APPROVED":
                    stats.auto_approved_count = count
                elif status == "EXPIRED":
                    stats.expired_count = count

            # Count approved/rejected today
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
            stats.approved_today = (
                self.db_session.query(func.count(DraftReply.id))
                .filter(
                    and_(
                        DraftReply.status.in_(["APPROVED", "SENT"]),
                        DraftReply.created_at >= today_start,
                    )
                )
                .scalar()
                or 0
            )

            stats.rejected_today = (
                self.db_session.query(func.count(DraftReply.id))
                .filter(
                    and_(
                        DraftReply.status.in_(["REJECTED", "DISCARDED"]),
                        DraftReply.created_at >= today_start,
                    )
                )
                .scalar()
                or 0
            )

            return stats

        except SQLAlchemyError as e:
            logger.error(f"Failed to get queue stats: {e}")
            return QueueStats()

    def set_auto_approve(self, enabled: bool) -> None:
        """Enable or disable auto-approval."""
        self._auto_approve_enabled = enabled
        logger.info(f"Auto-approve {'enabled' if enabled else 'disabled'}")

    def expire_old_drafts(self, hours: int = 72) -> int:
        """
        Expire PENDING drafts that have passed their expiration time.
        
        Phase 1.6.2: Uses expires_at field if set, otherwise falls back to
        created_at + hours for backwards compatibility with existing drafts.
        
        Args:
            hours: Fallback expiration age for drafts without expires_at (default 72h)
            
        Returns:
            Number of drafts marked as EXPIRED
        """
        try:
            from sqlalchemy import or_

            from core.database import DraftReply

            now = datetime.now(timezone.utc)
            fallback_cutoff = now - timedelta(hours=hours)
            
            # Expire drafts where:
            # 1. expires_at is set and has passed, OR
            # 2. expires_at is NULL and created_at is older than fallback
            count = (
                self.db_session.query(DraftReply)
                .filter(
                    DraftReply.status == "PENDING",
                    or_(
                        # New drafts with expires_at field
                        and_(
                            DraftReply.expires_at.isnot(None),
                            DraftReply.expires_at <= now,
                        ),
                        # Legacy drafts without expires_at (fallback to created_at)
                        and_(
                            DraftReply.expires_at.is_(None),
                            DraftReply.created_at < fallback_cutoff,
                        ),
                    ),
                )
                .update({"status": "EXPIRED"}, synchronize_session=False)
            )
            self.db_session.commit()

            if count > 0:
                logger.info("Expired %d stale PENDING drafts", count)
            return count

        except SQLAlchemyError as e:
            logger.error("Failed to expire old drafts: %s", e)
            self.db_session.rollback()
            return 0


# === MODULE TESTS ===


def module_tests() -> bool:
    """Run module-specific tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Approval Queue Service", "core/approval_queue.py")
    suite.start_suite()

    # Test 1: Enum values
    suite.run_test(
        "ApprovalStatus enum values",
        lambda: None
        if ApprovalStatus.PENDING.value == "PENDING"
        else (_ for _ in ()).throw(AssertionError("Wrong value")),
        test_summary="ApprovalStatus.PENDING has correct value",
        functions_tested="ApprovalStatus enum",
        method_description="Verify enum value mapping",
    )

    # Test 2: Priority enum
    suite.run_test(
        "ReviewPriority enum values",
        lambda: None
        if ReviewPriority.CRITICAL.value == "critical"
        else (_ for _ in ()).throw(AssertionError("Wrong value")),
        test_summary="ReviewPriority.CRITICAL has correct value",
        functions_tested="ReviewPriority enum",
        method_description="Verify enum value mapping",
    )

    # Test 3: QueueStats initialization
    def test_queue_stats() -> None:
        stats = QueueStats()
        assert stats.pending_count == 0, "pending_count should be 0"
        assert isinstance(stats.by_priority, dict), "by_priority should be dict"

    suite.run_test(
        "QueueStats initialization",
        test_queue_stats,
        test_summary="QueueStats initializes with correct defaults",
        functions_tested="QueueStats dataclass",
        method_description="Verify default values",
    )

    # Test 4: QueuedDraft dataclass
    def test_queued_draft() -> None:
        draft = QueuedDraft(
            draft_id=1,
            person_id=100,
            person_name="Test User",
            conversation_id="conv123",
            content="Test message",
            ai_confidence=85,
            priority=ReviewPriority.NORMAL,
            status=ApprovalStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        assert draft.draft_id == 1, "draft_id should be 1"
        assert draft.ai_confidence == 85, "ai_confidence should be 85"

    suite.run_test(
        "QueuedDraft dataclass",
        test_queued_draft,
        test_summary="QueuedDraft stores values correctly",
        functions_tested="QueuedDraft dataclass",
        method_description="Verify dataclass field storage",
    )

    # Test 5: ReviewDecision dataclass
    def test_review_decision() -> None:
        decision = ReviewDecision(
            success=True,
            draft_id=1,
            action="approve",
            message="Approved",
        )
        assert decision.success is True, "success should be True"
        assert decision.action == "approve", "action should be 'approve'"

    suite.run_test(
        "ReviewDecision dataclass",
        test_review_decision,
        test_summary="ReviewDecision stores values correctly",
        functions_tested="ReviewDecision dataclass",
        method_description="Verify dataclass field storage",
    )

    # Test 6: All status values are strings
    def test_status_values() -> None:
        for status in ApprovalStatus:
            assert isinstance(status.value, str), f"{status} value should be string"

    suite.run_test(
        "ApprovalStatus values are strings",
        test_status_values,
        test_summary="All ApprovalStatus values are strings",
        functions_tested="ApprovalStatus enum",
        method_description="Verify value types",
    )

    # Test 7: Reject updates state + records event (in-memory DB)
    def test_reject_updates_state_and_records_event() -> None:
        from core.database import ConversationState, DraftReply, EngagementTracking, Person
        from testing.test_utilities import create_test_database

        session = create_test_database()
        try:
            person = Person(username="Test User", profile_id="PROFILE_REJECT")
            session.add(person)
            session.commit()

            conv_state = ConversationState(people_id=person.id)
            session.add(conv_state)
            session.commit()

            draft = DraftReply(
                people_id=person.id,
                conversation_id="conv_reject_1",
                content="Draft content",
                status="PENDING",
            )
            session.add(draft)
            session.commit()

            svc = ApprovalQueueService(session)
            decision = svc.reject(draft.id, reviewer="tester", reason="not appropriate")
            assert decision.success is True

            updated = session.query(DraftReply).filter(DraftReply.id == draft.id).first()
            assert updated is not None
            assert updated.status == "REJECTED"

            updated_state = session.query(ConversationState).filter(ConversationState.people_id == person.id).first()
            assert updated_state is not None
            assert updated_state.next_action == "no_action"

            events = session.query(EngagementTracking).filter(EngagementTracking.people_id == person.id).all()
            assert any(e.event_type == "draft_rejected" for e in events), "Should record a draft_rejected event"
        finally:
            session.close()

    suite.run_test(
        "Reject updates state + records engagement",
        test_reject_updates_state_and_records_event,
        test_summary="Rejecting a draft marks it REJECTED, updates ConversationState.next_action, and records an engagement event.",
        functions_tested="ApprovalQueueService.reject",
        method_description="In-memory DB: Person + ConversationState + PENDING DraftReply then reject()",
    )

    # Test 8: Queue embeds review-only metadata (no schema migrations)
    def test_queue_embeds_internal_metadata() -> None:
        from core.database import DraftReply, Person
        from core.draft_content import strip_internal_metadata
        from testing.test_utilities import create_test_database

        session = create_test_database()
        try:
            person = Person(username="Test User", profile_id="PROFILE_QUEUE")
            session.add(person)
            session.commit()

            svc = ApprovalQueueService(session)
            draft_id_1 = svc.queue_for_review(
                person_id=person.id,
                conversation_id="conv_meta_1",
                content="Hello there",
                ai_confidence=88,
                _ai_reasoning="Reasoning (test)",
                _context_summary="Context (test)",
                _research_suggestions="Suggestion A\nSuggestion B",
                _research_metadata={"shared_match_count": 3},
            )
            assert draft_id_1 is not None

            stored = session.query(DraftReply).filter(DraftReply.id == draft_id_1).first()
            assert stored is not None
            assert "Hello there" in (stored.content or "")
            assert "research_suggestions" in (stored.content or "")
            assert strip_internal_metadata(stored.content or "") == "Hello there"

            # Updating should re-use the same pending draft row.
            draft_id_2 = svc.queue_for_review(
                person_id=person.id,
                conversation_id="conv_meta_1",
                content="Hello there",
                ai_confidence=88,
                _ai_reasoning="Reasoning updated (test)",
                _context_summary="Context updated (test)",
                _research_suggestions="Suggestion updated",
                _research_metadata={"shared_match_count": 5},
            )
            assert draft_id_2 == draft_id_1
        finally:
            session.close()

    suite.run_test(
        "Queue embeds internal draft metadata",
        test_queue_embeds_internal_metadata,
        test_summary="queue_for_review appends internal metadata into DraftReply.content and remains idempotent for a pending draft.",
        functions_tested="ApprovalQueueService.queue_for_review",
        method_description="In-memory DB: create Person, queue draft with _ai_reasoning/_context_summary, assert strip_internal_metadata returns clean message.",
    )

    # Test: expire_old_drafts uses expires_at field
    def test_expire_old_drafts_uses_expires_at() -> None:
        from core.database import DraftReply, Person
        from testing.test_utilities import create_test_database

        session = create_test_database()
        try:
            person = Person(username="Expiry Test", profile_id="PROFILE_EXPIRY")
            session.add(person)
            session.commit()

            now = datetime.now(timezone.utc)

            # Draft 1: expires_at in the past (should be expired)
            draft_expired = DraftReply(
                people_id=person.id,
                conversation_id="conv_exp_1",
                content="Old draft",
                status="PENDING",
                expires_at=now - timedelta(hours=1),
            )
            # Draft 2: expires_at in the future (should NOT be expired)
            draft_valid = DraftReply(
                people_id=person.id,
                conversation_id="conv_exp_2",
                content="Fresh draft",
                status="PENDING",
                expires_at=now + timedelta(hours=24),
            )
            # Draft 3: no expires_at, created recently (should NOT be expired by fallback)
            draft_legacy_fresh = DraftReply(
                people_id=person.id,
                conversation_id="conv_exp_3",
                content="Legacy fresh draft",
                status="PENDING",
                expires_at=None,
                created_at=now - timedelta(hours=1),
            )
            session.add_all([draft_expired, draft_valid, draft_legacy_fresh])
            session.commit()

            svc = ApprovalQueueService(session)
            expired_count = svc.expire_old_drafts(hours=72)

            # Only draft_expired should be marked EXPIRED
            assert expired_count == 1, f"Expected 1 expired, got {expired_count}"

            session.expire_all()
            assert draft_expired.status == "EXPIRED", "Past expires_at draft should be EXPIRED"
            assert draft_valid.status == "PENDING", "Future expires_at draft should stay PENDING"
            assert draft_legacy_fresh.status == "PENDING", "Recent legacy draft should stay PENDING"

        finally:
            session.close()

    suite.run_test(
        "expire_old_drafts uses expires_at field",
        test_expire_old_drafts_uses_expires_at,
        test_summary="expire_old_drafts correctly expires drafts based on expires_at field",
        functions_tested="ApprovalQueueService.expire_old_drafts",
        method_description="In-memory DB: create drafts with various expires_at values, verify only past-expiry drafts are marked EXPIRED",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run all tests with proper framework setup."""
    from testing.test_framework import create_standard_test_runner

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
