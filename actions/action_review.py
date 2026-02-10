"""
Review Queue Action Module.

Provides human-in-the-loop approval workflow for AI-proposed data updates.
Displays pending staged updates and data conflicts for user review,
allowing approval, rejection, or modification before changes are applied.

Features:
- List pending staged updates with visual diff
- Approve/reject individual or batch updates
- Conflict resolution workflow
- Summary statistics for review queue
"""


import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast

from sqlalchemy.orm import Session

if __package__ in {None, ""}:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from core.database import (
    ConflictStatusEnum,
    DataConflict,
    Person,
    StagedUpdate,
    UpdateStatusEnum,
)
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


class ReviewAction(Enum):
    """Actions that can be taken on a review item."""

    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"
    MODIFY = "modify"


@dataclass
class ReviewItem:
    """Unified view of an item requiring review."""

    id: int
    item_type: str  # 'staged_update' or 'data_conflict'
    person_id: int
    person_name: str
    field_name: str
    current_value: str | None
    proposed_value: str
    source: str
    confidence_score: int | None
    created_at: datetime

    @property
    def display_diff(self) -> str:
        """Generate a human-readable diff display."""
        current = self.current_value or "(empty)"
        return f"  Current:  {current}\n  Proposed: {self.proposed_value}"


@dataclass
class ReviewSummary:
    """Summary statistics for the review queue."""

    total_pending: int
    staged_updates: int
    data_conflicts: int
    by_field: dict[str, int]
    oldest_item_age_days: float
    avg_confidence_score: float | None


class ReviewQueue:
    """
    Manages the human-in-the-loop approval workflow for data updates.

    Provides methods to list, review, and resolve pending staged updates
    and data conflicts before they affect the database.
    """

    def __init__(self) -> None:
        """Initialize the review queue."""
        self.batch_size = 20

    @staticmethod
    def get_pending_items(
        db_session: Session,
        limit: int = 50,
        include_conflicts: bool = True,
    ) -> list[ReviewItem]:
        """
        Get all pending review items (staged updates and optionally conflicts).

        Args:
            db_session: Active database session
            limit: Maximum number of items to return
            include_conflicts: Whether to include data conflicts

        Returns:
            List of ReviewItem objects sorted by creation date
        """
        items: list[ReviewItem] = []

        # Get pending staged updates
        staged_updates = (
            db_session.query(StagedUpdate, Person)
            .join(Person, StagedUpdate.people_id == Person.id)
            .filter(StagedUpdate.status == UpdateStatusEnum.PENDING)
            .order_by(StagedUpdate.created_at.desc())
            .limit(limit)
            .all()
        )

        for update, person in staged_updates:
            items.append(
                ReviewItem(
                    id=update.id,
                    item_type="staged_update",
                    person_id=person.id,
                    person_name=person.first_name or person.username or f"Person #{person.id}",
                    field_name=update.field_name,
                    current_value=update.current_value,
                    proposed_value=update.proposed_value,
                    source=update.source,
                    confidence_score=update.confidence_score,
                    created_at=update.created_at,
                )
            )

        # Get open data conflicts
        if include_conflicts:
            conflicts = (
                db_session.query(DataConflict, Person)
                .join(Person, DataConflict.people_id == Person.id)
                .filter(DataConflict.status == ConflictStatusEnum.OPEN)
                .order_by(DataConflict.created_at.desc())
                .limit(limit)
                .all()
            )

            for conflict, person in conflicts:
                items.append(
                    ReviewItem(
                        id=conflict.id,
                        item_type="data_conflict",
                        person_id=person.id,
                        person_name=person.first_name or person.username or f"Person #{person.id}",
                        field_name=conflict.field_name,
                        current_value=conflict.existing_value,
                        proposed_value=conflict.new_value,
                        source=conflict.source,
                        confidence_score=conflict.confidence_score,
                        created_at=conflict.created_at,
                    )
                )

        # Sort by creation date (oldest first for review)
        items.sort(key=lambda x: x.created_at)
        return items[:limit]

    @staticmethod
    def get_summary(db_session: Session) -> ReviewSummary:
        """
        Get summary statistics for the review queue.

        Args:
            db_session: Active database session

        Returns:
            ReviewSummary with queue statistics
        """
        from sqlalchemy import func

        # Count staged updates
        staged_count = (
            db_session.query(func.count(StagedUpdate.id))
            .filter(StagedUpdate.status == UpdateStatusEnum.PENDING)
            .scalar()
            or 0
        )

        # Count data conflicts
        conflict_count = (
            db_session.query(func.count(DataConflict.id))
            .filter(DataConflict.status == ConflictStatusEnum.OPEN)
            .scalar()
            or 0
        )

        # Count by field (from both tables)
        field_counts: dict[str, int] = {}

        staged_fields = (
            db_session.query(StagedUpdate.field_name, func.count(StagedUpdate.id))
            .filter(StagedUpdate.status == UpdateStatusEnum.PENDING)
            .group_by(StagedUpdate.field_name)
            .all()
        )
        for field, count in staged_fields:
            field_counts[field] = field_counts.get(field, 0) + count

        conflict_fields = (
            db_session.query(DataConflict.field_name, func.count(DataConflict.id))
            .filter(DataConflict.status == ConflictStatusEnum.OPEN)
            .group_by(DataConflict.field_name)
            .all()
        )
        for field, count in conflict_fields:
            field_counts[field] = field_counts.get(field, 0) + count

        # Get oldest item age
        oldest_staged = (
            db_session.query(func.min(StagedUpdate.created_at))
            .filter(StagedUpdate.status == UpdateStatusEnum.PENDING)
            .scalar()
        )
        oldest_conflict = (
            db_session.query(func.min(DataConflict.created_at))
            .filter(DataConflict.status == ConflictStatusEnum.OPEN)
            .scalar()
        )

        oldest_date = None
        if oldest_staged and oldest_conflict:
            oldest_date = min(oldest_staged, oldest_conflict)
        elif oldest_staged:
            oldest_date = oldest_staged
        elif oldest_conflict:
            oldest_date = oldest_conflict

        age_days = 0.0
        if oldest_date:
            age_days = (datetime.now(UTC) - oldest_date).total_seconds() / 86400

        # Average confidence score
        avg_confidence = (
            db_session.query(func.avg(StagedUpdate.confidence_score))
            .filter(
                StagedUpdate.status == UpdateStatusEnum.PENDING,
                StagedUpdate.confidence_score.isnot(None),
            )
            .scalar()
        )

        return ReviewSummary(
            total_pending=staged_count + conflict_count,
            staged_updates=staged_count,
            data_conflicts=conflict_count,
            by_field=field_counts,
            oldest_item_age_days=age_days,
            avg_confidence_score=float(avg_confidence) if avg_confidence else None,
        )

    @staticmethod
    def approve_staged_update(
        db_session: Session,
        update_id: int,
        reviewer: str = "user",
        notes: str | None = None,
    ) -> bool:
        """
        Approve a staged update and apply changes to the Person record.

        Args:
            db_session: Active database session
            update_id: ID of the staged update to approve
            reviewer: Identifier of who approved
            notes: Optional review notes

        Returns:
            True if successful, False otherwise
        """
        update = db_session.query(StagedUpdate).filter(StagedUpdate.id == update_id).first()

        if not update:
            logger.warning(f"Staged update {update_id} not found")
            return False

        if update.status != UpdateStatusEnum.PENDING:
            logger.warning(f"Staged update {update_id} is not pending (status: {update.status.value})")
            return False

        # Get the person record
        person = db_session.query(Person).filter(Person.id == update.people_id).first()
        if not person:
            logger.error(f"Person {update.people_id} not found for staged update {update_id}")
            return False

        # Apply the update to the person
        if hasattr(person, update.field_name):
            old_value = getattr(person, update.field_name)
            setattr(person, update.field_name, update.proposed_value)
            logger.info(
                f"Applied update for person {person.id}: '{update.field_name}' {old_value} -> {update.proposed_value}"
            )

        # Update the staged update record
        now = datetime.now(UTC)
        update.status = UpdateStatusEnum.APPROVED
        update.reviewed_by = reviewer
        update.reviewer_notes = notes
        update.reviewed_at = now
        update.applied_at = now

        return True

    @staticmethod
    def reject_staged_update(
        db_session: Session,
        update_id: int,
        reviewer: str = "user",
        notes: str | None = None,
    ) -> bool:
        """
        Reject a staged update without applying changes.

        Args:
            db_session: Active database session
            update_id: ID of the staged update to reject
            reviewer: Identifier of who rejected
            notes: Optional rejection reason

        Returns:
            True if successful, False otherwise
        """
        update = db_session.query(StagedUpdate).filter(StagedUpdate.id == update_id).first()

        if not update:
            logger.warning(f"Staged update {update_id} not found")
            return False

        if update.status != UpdateStatusEnum.PENDING:
            logger.warning(f"Staged update {update_id} is not pending (status: {update.status.value})")
            return False

        # Update the record
        update.status = UpdateStatusEnum.REJECTED
        update.reviewed_by = reviewer
        update.reviewer_notes = notes
        update.reviewed_at = datetime.now(UTC)

        logger.info(f"Rejected staged update {update_id} for person {update.people_id}")
        return True

    @staticmethod
    def resolve_conflict(
        db_session: Session,
        conflict_id: int,
        accept_new: bool,
        reviewer: str = "user",
        notes: str | None = None,
    ) -> bool:
        """
        Resolve a data conflict by accepting or rejecting the new value.

        Args:
            db_session: Active database session
            conflict_id: ID of the data conflict to resolve
            accept_new: True to accept new value, False to keep existing
            reviewer: Identifier of who resolved
            notes: Optional resolution notes

        Returns:
            True if successful, False otherwise
        """
        conflict = db_session.query(DataConflict).filter(DataConflict.id == conflict_id).first()

        if not conflict:
            logger.warning(f"Data conflict {conflict_id} not found")
            return False

        if conflict.status != ConflictStatusEnum.OPEN:
            logger.warning(f"Data conflict {conflict_id} is not open (status: {conflict.status.value})")
            return False

        # Apply new value if accepted
        if accept_new:
            person = db_session.query(Person).filter(Person.id == conflict.people_id).first()
            if person and hasattr(person, conflict.field_name):
                old_value = getattr(person, conflict.field_name)
                setattr(person, conflict.field_name, conflict.new_value)
                logger.info(
                    f"Resolved conflict for person {person.id}: "
                    f"'{conflict.field_name}' {old_value} -> {conflict.new_value}"
                )

        # Update the conflict record
        conflict.status = ConflictStatusEnum.RESOLVED if accept_new else ConflictStatusEnum.REJECTED
        conflict.resolved_by = reviewer
        conflict.resolution_notes = notes
        conflict.resolved_at = datetime.now(UTC)

        return True

    @staticmethod
    def batch_expire_old_items(
        db_session: Session,
        max_age_days: int = 30,
    ) -> tuple[int, int]:
        """
        Expire old pending items that haven't been reviewed.

        Args:
            db_session: Active database session
            max_age_days: Maximum age in days before expiration

        Returns:
            Tuple of (expired_updates, expired_conflicts)
        """
        from sqlalchemy import func

        cutoff = datetime.now(UTC)
        cutoff = cutoff.replace(day=cutoff.day - max_age_days) if cutoff.day > max_age_days else cutoff

        # Expire old staged updates
        expired_updates = (
            db_session.query(StagedUpdate)
            .filter(
                StagedUpdate.status == UpdateStatusEnum.PENDING,
                StagedUpdate.created_at < cutoff,
            )
            .update(
                {
                    StagedUpdate.status: UpdateStatusEnum.EXPIRED,
                    StagedUpdate.reviewed_at: func.now(),
                    StagedUpdate.reviewer_notes: f"Auto-expired after {max_age_days} days",
                }
            )
        )

        # Expire old conflicts
        expired_conflicts = (
            db_session.query(DataConflict)
            .filter(
                DataConflict.status == ConflictStatusEnum.OPEN,
                DataConflict.created_at < cutoff,
            )
            .update(
                {
                    DataConflict.status: ConflictStatusEnum.ESCALATED,
                    DataConflict.resolved_at: func.now(),
                    DataConflict.resolution_notes: f"Auto-escalated after {max_age_days} days",
                }
            )
        )

        logger.info(f"Expired {expired_updates} staged updates, {expired_conflicts} conflicts")
        return expired_updates, expired_conflicts

    @staticmethod
    def display_item(item: ReviewItem) -> str:
        """
        Format a review item for display.

        Args:
            item: ReviewItem to display

        Returns:
            Formatted string for display
        """
        confidence = f" ({item.confidence_score}%)" if item.confidence_score else ""
        type_icon = "📝" if item.item_type == "staged_update" else "⚠️"
        return (
            f"{type_icon} [{item.id}] {item.person_name} - {item.field_name}{confidence}\n"
            f"   Source: {item.source}\n"
            f"{item.display_diff}"
        )

    @staticmethod
    def format_summary(summary: ReviewSummary) -> str:
        """
        Format review summary for display.

        Args:
            summary: ReviewSummary to format

        Returns:
            Formatted string for display
        """
        lines = [
            "📋 Review Queue Summary",
            "=" * 40,
            f"Total Pending:    {summary.total_pending}",
            f"  Staged Updates: {summary.staged_updates}",
            f"  Data Conflicts: {summary.data_conflicts}",
        ]

        if summary.by_field:
            lines.append("\nBy Field:")
            for field, count in sorted(summary.by_field.items(), key=lambda x: -x[1]):
                lines.append(f"  {field}: {count}")

        if summary.oldest_item_age_days > 0:
            lines.append(f"\nOldest Item: {summary.oldest_item_age_days:.1f} days")

        if summary.avg_confidence_score is not None:
            lines.append(f"Avg Confidence: {summary.avg_confidence_score:.1f}%")

        return "\n".join(lines)

    @staticmethod
    def approve_suggested_fact(
        db_session: Session,
        fact_id: int,
        reviewer: str = "user",  # noqa: ARG004 Reserved for audit logging
        apply_to_tree: bool = False,
        tree_id: str | None = None,
        session_manager: object | None = None,
    ) -> tuple[bool, str | None]:
        """
        Approve a suggested fact and optionally apply to Ancestry tree.

        Args:
            db_session: Active database session
            fact_id: ID of the SuggestedFact to approve
            reviewer: Identifier of who approved
            apply_to_tree: Whether to apply to Ancestry tree
            tree_id: Ancestry tree ID (required if apply_to_tree=True)
            session_manager: SessionManager instance (required if apply_to_tree=True)

        Returns:
            Tuple of (success, error_message)
        """
        from core.database import FactStatusEnum, SuggestedFact

        fact = db_session.query(SuggestedFact).filter(SuggestedFact.id == fact_id).first()

        if not fact:
            return False, f"SuggestedFact {fact_id} not found"

        if fact.status != FactStatusEnum.PENDING:
            return False, f"SuggestedFact {fact_id} is not pending (status: {fact.status.value})"

        # Update status to APPROVED
        fact.status = FactStatusEnum.APPROVED
        fact.updated_at = datetime.now(UTC)

        logger.info(f"Approved suggested fact {fact_id} for person {fact.people_id}")

        # Optionally apply to Ancestry tree
        if apply_to_tree:
            if not tree_id or not session_manager:
                return False, "tree_id and session_manager required for tree updates"

            from api.tree_update import TreeUpdateResult, TreeUpdateService
            from core.session_manager import SessionManager

            tree_service = TreeUpdateService(cast(SessionManager, session_manager))
            result = tree_service.apply_suggested_fact(db_session, fact, tree_id)

            if result.result != TreeUpdateResult.SUCCESS:
                return False, f"Tree update failed: {result.message}"

            logger.info(f"Applied fact {fact_id} to tree: {result.message}")

        return True, None

    @staticmethod
    def reject_suggested_fact(
        db_session: Session,
        fact_id: int,
        reviewer: str = "user",  # noqa: ARG004 Reserved for audit logging
        reason: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Reject a suggested fact without applying.

        Args:
            db_session: Active database session
            fact_id: ID of the SuggestedFact to reject
            reviewer: Identifier of who rejected
            reason: Optional rejection reason

        Returns:
            Tuple of (success, error_message)
        """
        from core.database import FactStatusEnum, SuggestedFact

        fact = db_session.query(SuggestedFact).filter(SuggestedFact.id == fact_id).first()

        if not fact:
            return False, f"SuggestedFact {fact_id} not found"

        if fact.status != FactStatusEnum.PENDING:
            return False, f"SuggestedFact {fact_id} is not pending (status: {fact.status.value})"

        fact.status = FactStatusEnum.REJECTED
        fact.updated_at = datetime.now(UTC)

        logger.info(f"Rejected suggested fact {fact_id} for person {fact.people_id}: {reason or 'no reason'}")
        return True, None

    @staticmethod
    def list_pending_suggested_facts(
        db_session: Session,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List pending suggested facts awaiting review.

        Args:
            db_session: Active database session
            limit: Maximum number of items to return

        Returns:
            List of pending SuggestedFacts as dictionaries
        """
        from core.database import FactStatusEnum, SuggestedFact

        facts = (
            db_session.query(SuggestedFact, Person)
            .join(Person, SuggestedFact.people_id == Person.id)
            .filter(SuggestedFact.status == FactStatusEnum.PENDING)
            .order_by(SuggestedFact.created_at.desc())
            .limit(limit)
            .all()
        )

        result: list[dict[str, Any]] = []
        for fact, person in facts:
            result.append(
                {
                    "id": fact.id,
                    "person_id": person.id,
                    "person_name": person.display_name,
                    "fact_type": fact.fact_type.value if fact.fact_type else "unknown",
                    "original_value": fact.original_value,
                    "new_value": fact.new_value,
                    "confidence_score": fact.confidence_score,
                    "created_at": fact.created_at.isoformat(),
                }
            )

        return result


# --- Test Functions ---


def _test_review_item_display_diff() -> None:
    """Test ReviewItem display_diff property."""
    item = ReviewItem(
        id=1,
        item_type="staged_update",
        person_id=100,
        person_name="John Smith",
        field_name="birth_year",
        current_value="1850",
        proposed_value="1852",
        source="conversation",
        confidence_score=85,
        created_at=datetime.now(UTC),
    )
    diff = item.display_diff
    assert "1850" in diff, "Should show current value"
    assert "1852" in diff, "Should show proposed value"


def _test_review_item_display_diff_empty() -> None:
    """Test ReviewItem display_diff with empty current value."""
    item = ReviewItem(
        id=2,
        item_type="staged_update",
        person_id=100,
        person_name="Jane Doe",
        field_name="death_year",
        current_value=None,
        proposed_value="1920",
        source="conversation",
        confidence_score=90,
        created_at=datetime.now(UTC),
    )
    diff = item.display_diff
    assert "(empty)" in diff, "Should show (empty) for None value"
    assert "1920" in diff, "Should show proposed value"


def _test_review_summary_creation() -> None:
    """Test ReviewSummary dataclass creation."""
    summary = ReviewSummary(
        total_pending=15,
        staged_updates=10,
        data_conflicts=5,
        by_field={"birth_year": 8, "birth_place": 7},
        oldest_item_age_days=3.5,
        avg_confidence_score=82.5,
    )
    assert summary.total_pending == 15, "Should have correct total"
    assert len(summary.by_field) == 2, "Should have 2 fields"


def _test_review_queue_init() -> None:
    """Test ReviewQueue initialization."""
    queue = ReviewQueue()
    assert queue.batch_size == 20, "Should have default batch size"


def _test_display_item_format() -> None:
    """Test ReviewQueue display_item formatting."""
    queue = ReviewQueue()
    item = ReviewItem(
        id=1,
        item_type="staged_update",
        person_id=100,
        person_name="Test Person",
        field_name="birth_year",
        current_value="1850",
        proposed_value="1852",
        source="conversation",
        confidence_score=85,
        created_at=datetime.now(UTC),
    )
    display = queue.display_item(item)
    assert "[1]" in display, "Should show item ID"
    assert "Test Person" in display, "Should show person name"
    assert "birth_year" in display, "Should show field name"
    assert "85%" in display, "Should show confidence"


def _test_display_item_conflict() -> None:
    """Test display_item for data conflict type."""
    queue = ReviewQueue()
    item = ReviewItem(
        id=99,
        item_type="data_conflict",
        person_id=200,
        person_name="Conflict Person",
        field_name="relationship",
        current_value="2nd Cousin",
        proposed_value="1st Cousin Once Removed",
        source="gedcom_import",
        confidence_score=None,
        created_at=datetime.now(UTC),
    )
    display = queue.display_item(item)
    assert "⚠️" in display, "Should show conflict icon"
    assert "[99]" in display, "Should show item ID"


def _test_format_summary() -> None:
    """Test ReviewQueue format_summary."""
    queue = ReviewQueue()
    summary = ReviewSummary(
        total_pending=10,
        staged_updates=7,
        data_conflicts=3,
        by_field={"birth_year": 5, "first_name": 5},
        oldest_item_age_days=2.5,
        avg_confidence_score=78.0,
    )
    formatted = queue.format_summary(summary)
    assert "Total Pending:" in formatted, "Should show total"
    assert "10" in formatted, "Should show count"
    assert "birth_year" in formatted, "Should show field breakdown"


def _test_review_action_enum() -> None:
    """Test ReviewAction enum values."""
    assert ReviewAction.APPROVE.value == "approve"
    assert ReviewAction.REJECT.value == "reject"
    assert ReviewAction.SKIP.value == "skip"
    assert ReviewAction.MODIFY.value == "modify"


# --- Test Suite Setup ---


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Review Queue Action", "actions/action_review.py")

    suite.run_test("Review item display diff", _test_review_item_display_diff)
    suite.run_test("Review item display diff empty", _test_review_item_display_diff_empty)
    suite.run_test("Review summary creation", _test_review_summary_creation)
    suite.run_test("Review queue initialization", _test_review_queue_init)
    suite.run_test("Display item formatting", _test_display_item_format)
    suite.run_test("Display item conflict type", _test_display_item_conflict)
    suite.run_test("Format summary", _test_format_summary)
    suite.run_test("Review action enum", _test_review_action_enum)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
