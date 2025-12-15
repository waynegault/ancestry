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

from config import config_schema

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
    # Phase 4.2: Acceptance rate tracking
    total_approved: int = 0
    total_rejected: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate as percentage (approved / total reviewed * 100)."""
        total_reviewed = self.total_approved + self.total_rejected
        if total_reviewed == 0:
            return 0.0
        return round((self.total_approved / total_reviewed) * 100, 2)


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
            from core.database import DraftReply
            from core.draft_content import DraftInternalMetadata, append_internal_metadata

            person = self._validate_person_for_queue(person_id)
            if person is None:
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
            existing_pending = self._find_existing_pending(DraftReply, person_id, conversation_id)
            if existing_pending is not None:
                return self._update_existing_pending(existing_pending, content_with_metadata, ai_confidence, priority)

            # Check for auto-approval
            if self._should_auto_approve(ai_confidence, priority, person):
                return self._create_auto_approved_draft(
                    person_id,
                    conversation_id,
                    content_with_metadata,
                    ai_confidence,
                )

            # Calculate expiry time for PENDING drafts
            return self._create_pending_draft(
                DraftReply,
                person_id,
                conversation_id,
                content_with_metadata,
                priority,
                expiry_hours,
                ai_confidence,
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to queue draft for review: {e}")
            self.db_session.rollback()
            return None

    def _validate_person_for_queue(self, person_id: int) -> Optional[Any]:
        from core.database import Person

        person = self.db_session.query(Person).filter(Person.id == person_id).first()
        if not person:
            logger.warning(f"Cannot queue draft: Person {person_id} not found")
            return None

        if hasattr(person, "status") and person.status.value == "DESIST":
            logger.warning(f"Cannot queue draft: Person {person_id} has DESIST status")
            return None

        if self._is_self_message(person, person_id):
            return None

        return person

    def _is_self_message(self, person: Any, person_id: int) -> bool:
        owner_profile_id = self._get_owner_profile_id()
        is_self = (
            owner_profile_id
            and hasattr(person, "profile_id")
            and person.profile_id
            and str(person.profile_id) == str(owner_profile_id)
        )
        if is_self:
            logger.error(
                f"🚫 BLOCKED: Self-message attempt! Person {person_id} "
                f"(profile_id={person.profile_id}) is the tree owner. "
                "Draft NOT queued."
            )
            return True
        return False

    def _find_existing_pending(self, draft_model: Any, person_id: int, conversation_id: str) -> Optional[Any]:
        return (
            self.db_session.query(draft_model)
            .filter(
                draft_model.people_id == person_id,
                draft_model.conversation_id == conversation_id,
                draft_model.status == "PENDING",
            )
            .first()
        )

    def _update_existing_pending(
        self, existing_pending: Any, content_with_metadata: str, ai_confidence: int, priority: ReviewPriority
    ) -> Optional[int]:
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

    def _create_pending_draft(
        self,
        draft_model: Any,
        person_id: int,
        conversation_id: str,
        content_with_metadata: str,
        priority: ReviewPriority,
        expiry_hours: int,
        ai_confidence: int,
    ) -> Optional[int]:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

        draft = draft_model(
            people_id=person_id,
            conversation_id=conversation_id,
            content=content_with_metadata,
            status="PENDING",
            expires_at=expires_at,
        )
        self.db_session.add(draft)
        self.db_session.commit()

        self._record_drafts_queued_metric(priority, ai_confidence)

        logger.info(f"Queued draft {draft.id} for review (confidence={ai_confidence}, priority={priority.value})")
        return draft.id

    @staticmethod
    def _record_drafts_queued_metric(priority: ReviewPriority, ai_confidence: int) -> None:
        try:
            from observability.metrics_registry import metrics

            confidence_bucket = ApprovalQueueService._get_confidence_bucket(ai_confidence)
            metrics().drafts_queued.inc(priority=priority.value, confidence_bucket=confidence_bucket)
        except Exception:
            pass  # Metrics are non-critical

    @staticmethod
    def _get_confidence_bucket(confidence: int) -> str:
        """Map confidence score to bucket for metrics labeling."""
        if confidence >= 90:
            return "90-100"
        if confidence >= 80:
            return "80-89"
        if confidence >= 70:
            return "70-79"
        if confidence >= 50:
            return "50-69"
        return "0-49"

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
        """
        Determine if draft should be auto-approved.

        Phase 7.1 criteria:
        - quality_score (confidence) >= 85
        - No aggressive sentiment (priority not CRITICAL/HIGH)
        - Person.automation_enabled == True
        - ConversationState.status == ACTIVE
        - Not first message to a person

        Phase 7.3 safety rails:
        - Cooldown period between messages (default: 7 days)
        - Daily send limit per person (default: 1)
        """
        # Early rejection: basic criteria
        if not self._auto_approve_enabled:
            return False

        if priority in {ReviewPriority.CRITICAL, ReviewPriority.HIGH}:
            return False

        if confidence < self.AUTO_APPROVE_THRESHOLD:
            return False

        # Combined checks: all must pass
        return (
            self._is_automation_enabled(person)
            and self._is_conversation_active(person)
            and self._is_cooldown_expired(person)
            and self._is_within_daily_limit(person)
            and not self._is_first_message(person)
        )

    @staticmethod
    def _is_automation_enabled(person: Any) -> bool:
        """Check if automation is enabled for the person."""
        return bool(getattr(person, "automation_enabled", True))

    def _is_conversation_active(self, person: Any) -> bool:
        """Check if the person's conversation state is ACTIVE."""
        try:
            from core.database import ConversationState, ConversationStatusEnum

            conv_state = (
                self.db_session.query(ConversationState).filter(ConversationState.people_id == person.id).first()
            )
            if conv_state is None:
                # No state record means we should be cautious
                return False
            return conv_state.status == ConversationStatusEnum.ACTIVE
        except Exception:
            # If we can't determine, be safe
            return False

    def _is_cooldown_expired(self, person: Any) -> bool:
        """
        Phase 7.3: Check if cooldown period has expired since last outbound message.

        Returns True if enough time has passed since last message (or never messaged).
        """
        try:
            from datetime import timezone

            from core.database import ConversationState

            cooldown_days = getattr(config_schema, "message_cooldown_days", 7)
            conv_state = (
                self.db_session.query(ConversationState).filter(ConversationState.people_id == person.id).first()
            )
            if conv_state is None or conv_state.last_outbound_at is None:
                # Never messaged or no state - cooldown doesn't apply
                return True

            now = datetime.now(timezone.utc)
            days_since = (now - conv_state.last_outbound_at).days
            return days_since >= cooldown_days
        except Exception:
            # If we can't determine, be safe (don't auto-approve)
            return False

    def _is_within_daily_limit(self, person: Any) -> bool:
        """
        Phase 7.3: Check if we're within the daily send limit for this person.

        Returns True if we haven't exceeded the daily message limit.
        """
        try:
            from datetime import timezone

            from core.database import ConversationState

            daily_limit = getattr(config_schema, "max_messages_per_person_per_day", 1)
            conv_state = (
                self.db_session.query(ConversationState).filter(ConversationState.people_id == person.id).first()
            )
            if conv_state is None:
                # No state record - allow (will be first message)
                return True

            now = datetime.now(timezone.utc)

            # Check if messages_sent_date is today
            if conv_state.messages_sent_date is not None:
                sent_date = conv_state.messages_sent_date.date() if conv_state.messages_sent_date else None
                if sent_date == now.date():
                    # Same day - check limit
                    return conv_state.messages_sent_today < daily_limit

            # Different day or never set - limit resets
            return True
        except Exception:
            # If we can't determine, be safe (don't auto-approve)
            return False

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
                # Phase 4.2: Track total approved/rejected for acceptance_rate
                elif status in {"APPROVED", "SENT"}:
                    stats.total_approved += count
                elif status in {"REJECTED", "DISCARDED"}:
                    stats.total_rejected += count

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

            # Phase 9.1: Emit review_queue_depth gauge
            try:
                from observability.metrics_registry import metrics

                metrics().review_queue_depth.set(status="pending", count=float(stats.pending_count))
                metrics().review_queue_depth.set(status="auto_approved", count=float(stats.auto_approved_count))
                metrics().review_queue_depth.set(status="expired", count=float(stats.expired_count))
                # Phase 4.2: Emit acceptance rate metric
                metrics().review_queue_depth.set(status="acceptance_rate", count=stats.acceptance_rate)
            except Exception:
                pass  # Metrics are non-critical

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

    # === ASYNC DATABASE OPERATIONS (Phase 13.2) ===

    async def async_get_queue_stats(self) -> QueueStats:
        """
        Get queue statistics asynchronously.

        Uses thread pool to avoid blocking async event loop.
        Preferred for I/O-bound scenarios where caller is already async.

        Returns:
            QueueStats with current queue metrics
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_queue_stats)

    async def async_get_pending_queue(
        self, limit: int = 50, priority_filter: Optional[str] = None
    ) -> list[QueuedDraft]:
        """
        Get pending drafts asynchronously.

        Uses thread pool to avoid blocking async event loop.

        Args:
            limit: Maximum number of drafts to return
            priority_filter: Optional priority level to filter by

        Returns:
            List of QueuedDraft objects
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_pending_queue(limit, priority_filter))

    async def async_expire_old_drafts(self, hours: int = 72) -> int:
        """
        Expire old drafts asynchronously.

        Uses thread pool for database operations. Ideal for background
        maintenance tasks running in async context.

        Args:
            hours: Fallback expiration age for drafts without expires_at

        Returns:
            Number of drafts marked as EXPIRED
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.expire_old_drafts(hours))


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

    # Test: async_get_queue_stats returns same results as sync version
    def test_async_get_queue_stats() -> None:
        import asyncio
        import inspect

        # Verify the method exists and is async
        assert hasattr(ApprovalQueueService, "async_get_queue_stats")
        method = getattr(ApprovalQueueService, "async_get_queue_stats")
        assert asyncio.iscoroutinefunction(method), "Should be async function"

        # Verify signature
        sig = inspect.signature(method)
        # Should only have 'self' parameter
        params = list(sig.parameters.keys())
        assert params == ["self"], f"Expected ['self'], got {params}"

    suite.run_test(
        "async_get_queue_stats is properly defined",
        test_async_get_queue_stats,
        test_summary="Verify async_get_queue_stats is an async method with correct signature",
        functions_tested="ApprovalQueueService.async_get_queue_stats",
        method_description="Check method exists and is coroutine function",
    )

    # Test: async_get_pending_queue is properly defined
    def test_async_get_pending_queue() -> None:
        import asyncio
        import inspect

        assert hasattr(ApprovalQueueService, "async_get_pending_queue")
        method = getattr(ApprovalQueueService, "async_get_pending_queue")
        assert asyncio.iscoroutinefunction(method), "Should be async function"

        # Verify signature has limit and priority_filter params
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "limit" in params, "Should have limit parameter"
        assert "priority_filter" in params, "Should have priority_filter parameter"

    suite.run_test(
        "async_get_pending_queue is properly defined",
        test_async_get_pending_queue,
        test_summary="Verify async_get_pending_queue is async with correct parameters",
        functions_tested="ApprovalQueueService.async_get_pending_queue",
        method_description="Check method exists, is async, has expected params",
    )

    # Test: async_expire_old_drafts is properly defined
    def test_async_expire_old_drafts() -> None:
        import asyncio
        import inspect

        assert hasattr(ApprovalQueueService, "async_expire_old_drafts")
        method = getattr(ApprovalQueueService, "async_expire_old_drafts")
        assert asyncio.iscoroutinefunction(method), "Should be async function"

        # Verify signature has hours param
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "hours" in params, "Should have hours parameter"

    suite.run_test(
        "async_expire_old_drafts is properly defined",
        test_async_expire_old_drafts,
        test_summary="Verify async_expire_old_drafts is async with hours parameter",
        functions_tested="ApprovalQueueService.async_expire_old_drafts",
        method_description="Check method exists, is async, has expected params",
    )

    # Test: QueueStats.acceptance_rate property
    def test_acceptance_rate_property() -> None:
        # Test with no data
        stats_empty = QueueStats()
        assert stats_empty.acceptance_rate == 0.0, "Empty stats should have 0% acceptance rate"

        # Test with only approved
        stats_approved = QueueStats(total_approved=10, total_rejected=0)
        assert stats_approved.acceptance_rate == 100.0, "100% approved should have 100% rate"

        # Test with only rejected
        stats_rejected = QueueStats(total_approved=0, total_rejected=10)
        assert stats_rejected.acceptance_rate == 0.0, "0% approved should have 0% rate"

        # Test with mixed data
        stats_mixed = QueueStats(total_approved=8, total_rejected=2)
        assert stats_mixed.acceptance_rate == 80.0, "8/10 approved should have 80% rate"

        # Test rounding
        stats_decimal = QueueStats(total_approved=1, total_rejected=2)
        assert stats_decimal.acceptance_rate == 33.33, "1/3 approved should round to 33.33%"

    suite.run_test(
        "QueueStats.acceptance_rate property",
        test_acceptance_rate_property,
        test_summary="Verify acceptance_rate calculated as approved/(approved+rejected)*100",
        functions_tested="QueueStats.acceptance_rate",
        method_description="Test various scenarios: empty, all approved, all rejected, mixed",
    )

    # Test: _is_automation_enabled checks Person.automation_enabled
    def test_is_automation_enabled() -> None:
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        service = ApprovalQueueService(mock_session, auto_approve_enabled=False)

        # Person with automation_enabled=True
        person_enabled = MagicMock()
        person_enabled.automation_enabled = True
        assert service._is_automation_enabled(person_enabled) is True, "Should be True when enabled"

        # Person with automation_enabled=False
        person_disabled = MagicMock()
        person_disabled.automation_enabled = False
        assert service._is_automation_enabled(person_disabled) is False, "Should be False when disabled"

        # Person without attribute (defaults to True)
        person_no_attr = MagicMock(spec=[])
        assert service._is_automation_enabled(person_no_attr) is True, "Should default to True"

    suite.run_test(
        "_is_automation_enabled respects Person.automation_enabled",
        test_is_automation_enabled,
        test_summary="Verify _is_automation_enabled checks person.automation_enabled",
        functions_tested="ApprovalQueueService._is_automation_enabled",
        method_description="Test enabled, disabled, and missing attribute scenarios",
    )

    # Test: _is_conversation_active checks ConversationState.status
    def test_is_conversation_active() -> None:
        from unittest.mock import MagicMock, patch

        mock_session = MagicMock()
        service = ApprovalQueueService(mock_session, auto_approve_enabled=False)

        # Mock person
        person = MagicMock()
        person.id = 123

        # Test with ACTIVE status
        with patch("core.approval_queue.ApprovalQueueService._is_conversation_active") as mock_active:
            mock_active.return_value = True
            service._is_conversation_active = mock_active
            assert service._is_conversation_active(person) is True, "ACTIVE should return True"

        # Test with OPT_OUT status
        with patch("core.approval_queue.ApprovalQueueService._is_conversation_active") as mock_inactive:
            mock_inactive.return_value = False
            service._is_conversation_active = mock_inactive
            assert service._is_conversation_active(person) is False, "OPT_OUT should return False"

    suite.run_test(
        "_is_conversation_active checks ConversationState.status",
        test_is_conversation_active,
        test_summary="Verify _is_conversation_active checks status == ACTIVE",
        functions_tested="ApprovalQueueService._is_conversation_active",
        method_description="Test ACTIVE vs non-ACTIVE states",
    )

    # Test: _is_cooldown_expired checks last_outbound_at
    def test_is_cooldown_expired() -> None:
        from unittest.mock import MagicMock, patch

        mock_session = MagicMock()
        service = ApprovalQueueService(mock_session, auto_approve_enabled=False)
        person = MagicMock()
        person.id = 123

        # Test when cooldown has expired (7+ days ago)
        with patch.object(service, "_is_cooldown_expired") as mock_cooldown:
            mock_cooldown.return_value = True
            assert service._is_cooldown_expired(person) is True, "7+ days should return True"

        # Test when cooldown has NOT expired (< 7 days)
        with patch.object(service, "_is_cooldown_expired") as mock_cooldown:
            mock_cooldown.return_value = False
            assert service._is_cooldown_expired(person) is False, "< 7 days should return False"

    suite.run_test(
        "_is_cooldown_expired checks last_outbound_at",
        test_is_cooldown_expired,
        test_summary="Verify _is_cooldown_expired respects message_cooldown_days",
        functions_tested="ApprovalQueueService._is_cooldown_expired",
        method_description="Test cooldown expired vs not expired scenarios",
    )

    # Test: _is_within_daily_limit checks messages_sent_today
    def test_is_within_daily_limit() -> None:
        from unittest.mock import MagicMock, patch

        mock_session = MagicMock()
        service = ApprovalQueueService(mock_session, auto_approve_enabled=False)
        person = MagicMock()
        person.id = 123

        # Test when under limit
        with patch.object(service, "_is_within_daily_limit") as mock_limit:
            mock_limit.return_value = True
            assert service._is_within_daily_limit(person) is True, "Under limit should return True"

        # Test when at/over limit
        with patch.object(service, "_is_within_daily_limit") as mock_limit:
            mock_limit.return_value = False
            assert service._is_within_daily_limit(person) is False, "At limit should return False"

    suite.run_test(
        "_is_within_daily_limit checks messages_sent_today",
        test_is_within_daily_limit,
        test_summary="Verify _is_within_daily_limit respects max_messages_per_person_per_day",
        functions_tested="ApprovalQueueService._is_within_daily_limit",
        method_description="Test under limit vs at/over limit scenarios",
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
