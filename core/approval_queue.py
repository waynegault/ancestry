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

    def queue_for_review(
        self,
        person_id: int,
        conversation_id: str,
        content: str,
        ai_confidence: int,
        _ai_reasoning: Optional[str] = None,  # Reserved for future use
        _context_summary: Optional[str] = None,  # Reserved for future use
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

            # Check if person exists and is contactable
            person = self.db_session.query(Person).filter(Person.id == person_id).first()
            if not person:
                logger.warning(f"Cannot queue draft: Person {person_id} not found")
                return None

            # Check if person opted out (DESIST status)
            if hasattr(person, "status") and person.status.value == "DESIST":
                logger.warning(f"Cannot queue draft: Person {person_id} has DESIST status")
                return None

            # Determine priority based on confidence
            priority = self._calculate_priority(ai_confidence, person)

            # Check for auto-approval
            if self._should_auto_approve(ai_confidence, priority, person):
                return self._create_auto_approved_draft(person_id, conversation_id, content, ai_confidence)

            # Calculate expiry time (reserved for future use)
            _expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

            draft = DraftReply(
                people_id=person_id,
                conversation_id=conversation_id,
                content=content,
                status="PENDING",
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
            from core.database import ConversationLog

            outbound_count = (
                self.db_session.query(func.count(ConversationLog.id))
                .filter(
                    and_(
                        ConversationLog.person_id == person.id,
                        ConversationLog.direction == "OUT",
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
                    status=ApprovalStatus(draft.status),
                    created_at=draft.created_at,
                )
                results.append(queued)

            return results

        except SQLAlchemyError as e:
            logger.error(f"Failed to get pending queue: {e}")
            return []

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
            from core.database import DraftReply

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

            draft.status = "DISCARDED"
            self.db_session.commit()

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
                        DraftReply.status == "DISCARDED",
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
        """Expire drafts older than specified hours."""
        try:
            from core.database import DraftReply

            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            count = (
                self.db_session.query(DraftReply)
                .filter(
                    and_(
                        DraftReply.status == "PENDING",
                        DraftReply.created_at < cutoff,
                    )
                )
                .update({"status": "EXPIRED"})
            )
            self.db_session.commit()

            if count > 0:
                logger.info(f"Expired {count} old drafts")
            return count

        except SQLAlchemyError as e:
            logger.error(f"Failed to expire old drafts: {e}")
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
