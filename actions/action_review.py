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
from datetime import UTC, datetime, timedelta
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
    FactStatusEnum,
    FactTypeEnum,
    Person,
    StagedUpdate,
    SuggestedFact,
    UpdateStatusEnum,
)
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner, create_test_database

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

        # SQLite returns naive datetimes; normalise both to UTC-aware before comparing
        if oldest_staged and oldest_staged.tzinfo is None:
            oldest_staged = oldest_staged.replace(tzinfo=UTC)
        if oldest_conflict and oldest_conflict.tzinfo is None:
            oldest_conflict = oldest_conflict.replace(tzinfo=UTC)

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

        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)

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


def _test_approve_staged_update_db() -> None:
    """Test approving a staged update applies changes to Person record."""
    session = create_test_database()
    person = Person(username="John Smith", first_name="John", birth_year=1850)
    session.add(person)
    session.commit()

    update = StagedUpdate(
        people_id=person.id,
        field_name="birth_year",
        current_value="1850",
        proposed_value="1852",
        source="conversation",
        confidence_score=85,
        status=UpdateStatusEnum.PENDING,
    )
    session.add(update)
    session.commit()
    update_id = update.id

    queue = ReviewQueue()
    result = queue.approve_staged_update(session, update_id, reviewer="tester", notes="Verified")
    session.commit()

    assert result is True, "Approval should succeed"
    refreshed_update = session.query(StagedUpdate).filter(StagedUpdate.id == update_id).first()
    assert refreshed_update.status == UpdateStatusEnum.APPROVED, "Status should be APPROVED"
    assert refreshed_update.reviewed_by == "tester", "Reviewer should be recorded"
    assert refreshed_update.reviewer_notes == "Verified", "Notes should be recorded"
    assert refreshed_update.reviewed_at is not None, "Reviewed timestamp should be set"
    assert refreshed_update.applied_at is not None, "Applied timestamp should be set"

    refreshed_person = session.query(Person).filter(Person.id == person.id).first()
    assert str(refreshed_person.birth_year) == "1852" or refreshed_person.birth_year == "1852", \
        "Person birth_year should be updated to proposed value"
    session.close()


def _test_approve_staged_update_not_found() -> None:
    """Test approving a non-existent staged update returns False."""
    session = create_test_database()
    queue = ReviewQueue()
    result = queue.approve_staged_update(session, 9999)
    assert result is False, "Should return False for non-existent update"
    session.close()


def _test_approve_already_approved() -> None:
    """Test approving an already-approved staged update returns False."""
    session = create_test_database()
    person = Person(username="Already Done")
    session.add(person)
    session.commit()

    update = StagedUpdate(
        people_id=person.id,
        field_name="first_name",
        current_value=None,
        proposed_value="Jane",
        source="conversation",
        status=UpdateStatusEnum.APPROVED,
    )
    session.add(update)
    session.commit()

    queue = ReviewQueue()
    result = queue.approve_staged_update(session, update.id)
    assert result is False, "Should return False for non-PENDING update"
    session.close()


def _test_reject_staged_update_db() -> None:
    """Test rejecting a staged update sets status without applying changes."""
    session = create_test_database()
    person = Person(username="Bob Jones", first_name="Bob", birth_year=1900)
    session.add(person)
    session.commit()

    update = StagedUpdate(
        people_id=person.id,
        field_name="birth_year",
        current_value="1900",
        proposed_value="1905",
        source="conversation",
        confidence_score=60,
        status=UpdateStatusEnum.PENDING,
    )
    session.add(update)
    session.commit()
    update_id = update.id

    queue = ReviewQueue()
    result = queue.reject_staged_update(session, update_id, reviewer="reviewer1", notes="Incorrect")
    session.commit()

    assert result is True, "Rejection should succeed"
    refreshed = session.query(StagedUpdate).filter(StagedUpdate.id == update_id).first()
    assert refreshed.status == UpdateStatusEnum.REJECTED, "Status should be REJECTED"
    assert refreshed.reviewed_by == "reviewer1", "Reviewer should be recorded"
    assert refreshed.reviewer_notes == "Incorrect", "Notes should be recorded"
    assert refreshed.reviewed_at is not None, "Reviewed timestamp should be set"

    refreshed_person = session.query(Person).filter(Person.id == person.id).first()
    assert refreshed_person.birth_year == 1900, "Person birth_year should NOT change on rejection"
    session.close()


def _test_reject_staged_update_not_found() -> None:
    """Test rejecting a non-existent staged update returns False."""
    session = create_test_database()
    queue = ReviewQueue()
    result = queue.reject_staged_update(session, 9999)
    assert result is False, "Should return False for non-existent update"
    session.close()


def _test_resolve_conflict_accept() -> None:
    """Test resolving a conflict by accepting the new value."""
    session = create_test_database()
    person = Person(username="Conflict Test", first_name="Alice")
    session.add(person)
    session.commit()

    conflict = DataConflict(
        people_id=person.id,
        field_name="first_name",
        existing_value="Alice",
        new_value="Alicia",
        source="gedcom_import",
        status=ConflictStatusEnum.OPEN,
    )
    session.add(conflict)
    session.commit()
    conflict_id = conflict.id

    queue = ReviewQueue()
    result = queue.resolve_conflict(session, conflict_id, accept_new=True, reviewer="tester")
    session.commit()

    assert result is True, "Resolve should succeed"
    refreshed_conflict = session.query(DataConflict).filter(DataConflict.id == conflict_id).first()
    assert refreshed_conflict.status == ConflictStatusEnum.RESOLVED, "Status should be RESOLVED"
    assert refreshed_conflict.resolved_by == "tester", "Resolver should be recorded"
    assert refreshed_conflict.resolved_at is not None, "Resolved timestamp should be set"

    refreshed_person = session.query(Person).filter(Person.id == person.id).first()
    assert refreshed_person.first_name == "Alicia", "Person first_name should be updated to new value"
    session.close()


def _test_resolve_conflict_reject() -> None:
    """Test resolving a conflict by rejecting the new value keeps existing."""
    session = create_test_database()
    person = Person(username="Conflict Test 2", first_name="Bob")
    session.add(person)
    session.commit()

    conflict = DataConflict(
        people_id=person.id,
        field_name="first_name",
        existing_value="Bob",
        new_value="Robert",
        source="api_sync",
        status=ConflictStatusEnum.OPEN,
    )
    session.add(conflict)
    session.commit()
    conflict_id = conflict.id

    queue = ReviewQueue()
    result = queue.resolve_conflict(session, conflict_id, accept_new=False, reviewer="tester")
    session.commit()

    assert result is True, "Resolve should succeed"
    refreshed_conflict = session.query(DataConflict).filter(DataConflict.id == conflict_id).first()
    assert refreshed_conflict.status == ConflictStatusEnum.REJECTED, "Status should be REJECTED"

    refreshed_person = session.query(Person).filter(Person.id == person.id).first()
    assert refreshed_person.first_name == "Bob", "Person first_name should remain unchanged"
    session.close()


def _test_get_pending_items_db() -> None:
    """Test get_pending_items returns staged updates and conflicts from DB."""
    session = create_test_database()
    person = Person(username="PendingTest", first_name="Charlie")
    session.add(person)
    session.commit()

    update = StagedUpdate(
        people_id=person.id,
        field_name="birth_year",
        current_value=None,
        proposed_value="1880",
        source="conversation",
        confidence_score=90,
        status=UpdateStatusEnum.PENDING,
    )
    conflict = DataConflict(
        people_id=person.id,
        field_name="first_name",
        existing_value="Charlie",
        new_value="Charles",
        source="gedcom_import",
        status=ConflictStatusEnum.OPEN,
    )
    session.add_all([update, conflict])
    session.commit()

    queue = ReviewQueue()
    items = queue.get_pending_items(session)

    assert len(items) == 2, f"Should have 2 pending items, got {len(items)}"
    item_types = {i.item_type for i in items}
    assert "staged_update" in item_types, "Should include staged_update"
    assert "data_conflict" in item_types, "Should include data_conflict"
    session.close()


def _test_get_pending_items_excludes_non_pending() -> None:
    """Test get_pending_items excludes approved/rejected items."""
    session = create_test_database()
    person = Person(username="FilterTest", first_name="Diana")
    session.add(person)
    session.commit()

    pending = StagedUpdate(
        people_id=person.id,
        field_name="birth_year",
        proposed_value="1870",
        source="conversation",
        status=UpdateStatusEnum.PENDING,
    )
    approved = StagedUpdate(
        people_id=person.id,
        field_name="first_name",
        proposed_value="Diane",
        source="conversation",
        status=UpdateStatusEnum.APPROVED,
    )
    session.add_all([pending, approved])
    session.commit()

    queue = ReviewQueue()
    items = queue.get_pending_items(session, include_conflicts=False)

    assert len(items) == 1, f"Should only have 1 pending item, got {len(items)}"
    assert items[0].field_name == "birth_year", "Should only return the PENDING update"
    session.close()


def _test_get_summary_db() -> None:
    """Test get_summary returns correct counts from DB."""
    session = create_test_database()
    person = Person(username="SummaryTest", first_name="Eve")
    session.add(person)
    session.commit()

    updates = [
        StagedUpdate(
            people_id=person.id, field_name="birth_year", proposed_value="1870",
            source="conversation", confidence_score=80, status=UpdateStatusEnum.PENDING,
        ),
        StagedUpdate(
            people_id=person.id, field_name="birth_year", proposed_value="1871",
            source="conversation", confidence_score=90, status=UpdateStatusEnum.PENDING,
        ),
    ]
    conflict = DataConflict(
        people_id=person.id, field_name="first_name",
        existing_value="Eve", new_value="Eva",
        source="gedcom_import", status=ConflictStatusEnum.OPEN,
    )
    session.add_all([*updates, conflict])
    session.commit()

    queue = ReviewQueue()
    summary = queue.get_summary(session)

    assert summary.staged_updates == 2, f"Should have 2 staged updates, got {summary.staged_updates}"
    assert summary.data_conflicts == 1, f"Should have 1 data conflict, got {summary.data_conflicts}"
    assert summary.total_pending == 3, f"Should have 3 total pending, got {summary.total_pending}"
    assert "birth_year" in summary.by_field, "Should have birth_year in field breakdown"
    assert summary.by_field["birth_year"] == 2, "Should count 2 birth_year updates"
    assert summary.avg_confidence_score is not None, "Should compute avg confidence"
    assert 84.0 <= summary.avg_confidence_score <= 86.0, "Avg confidence should be ~85"
    session.close()


def _test_approve_suggested_fact_db() -> None:
    """Test approving a SuggestedFact sets status to APPROVED."""
    session = create_test_database()
    person = Person(username="FactTest", first_name="Frank")
    session.add(person)
    session.commit()

    fact = SuggestedFact(
        people_id=person.id,
        fact_type=FactTypeEnum.BIRTH,
        original_value="born about 1850",
        new_value="1850",
        confidence_score=75,
        status=FactStatusEnum.PENDING,
    )
    session.add(fact)
    session.commit()
    fact_id = fact.id

    queue = ReviewQueue()
    success, error = queue.approve_suggested_fact(session, fact_id, reviewer="tester")
    session.commit()

    assert success is True, f"Approval should succeed, got error: {error}"
    assert error is None, "Error should be None on success"
    refreshed = session.query(SuggestedFact).filter(SuggestedFact.id == fact_id).first()
    assert refreshed.status == FactStatusEnum.APPROVED, "Status should be APPROVED"
    assert refreshed.updated_at is not None, "Updated timestamp should be set"
    session.close()


def _test_approve_suggested_fact_not_found() -> None:
    """Test approving a non-existent SuggestedFact returns failure."""
    session = create_test_database()
    queue = ReviewQueue()
    success, error = queue.approve_suggested_fact(session, 9999)
    assert success is False, "Should fail for non-existent fact"
    assert error is not None, "Should return error message"
    assert "not found" in error.lower(), "Error should mention not found"
    session.close()


def _test_reject_suggested_fact_db() -> None:
    """Test rejecting a SuggestedFact sets status to REJECTED."""
    session = create_test_database()
    person = Person(username="RejectFactTest", first_name="Grace")
    session.add(person)
    session.commit()

    fact = SuggestedFact(
        people_id=person.id,
        fact_type=FactTypeEnum.DEATH,
        original_value="died 1920",
        new_value="1920",
        confidence_score=65,
        status=FactStatusEnum.PENDING,
    )
    session.add(fact)
    session.commit()
    fact_id = fact.id

    queue = ReviewQueue()
    success, error = queue.reject_suggested_fact(session, fact_id, reviewer="tester", reason="Unverified")
    session.commit()

    assert success is True, f"Rejection should succeed, got error: {error}"
    refreshed = session.query(SuggestedFact).filter(SuggestedFact.id == fact_id).first()
    assert refreshed.status == FactStatusEnum.REJECTED, "Status should be REJECTED"
    session.close()


def _test_reject_suggested_fact_already_approved() -> None:
    """Test rejecting an already-approved SuggestedFact returns failure."""
    session = create_test_database()
    person = Person(username="AlreadyApproved")
    session.add(person)
    session.commit()

    fact = SuggestedFact(
        people_id=person.id,
        fact_type=FactTypeEnum.LOCATION,
        original_value="New York",
        new_value="New York, NY",
        status=FactStatusEnum.APPROVED,
    )
    session.add(fact)
    session.commit()

    queue = ReviewQueue()
    success, error = queue.reject_suggested_fact(session, fact.id)
    assert success is False, "Should fail for non-PENDING fact"
    assert error is not None and "not pending" in error.lower(), "Error should mention not pending"
    session.close()


def _test_list_pending_suggested_facts_db() -> None:
    """Test list_pending_suggested_facts returns only PENDING facts."""
    session = create_test_database()
    person = Person(username="ListTest", first_name="Hank")
    session.add(person)
    session.commit()

    pending_fact = SuggestedFact(
        people_id=person.id,
        fact_type=FactTypeEnum.BIRTH,
        original_value="born 1860",
        new_value="1860",
        confidence_score=80,
        status=FactStatusEnum.PENDING,
    )
    approved_fact = SuggestedFact(
        people_id=person.id,
        fact_type=FactTypeEnum.DEATH,
        original_value="died 1930",
        new_value="1930",
        confidence_score=90,
        status=FactStatusEnum.APPROVED,
    )
    session.add_all([pending_fact, approved_fact])
    session.commit()

    queue = ReviewQueue()
    results = queue.list_pending_suggested_facts(session)

    assert len(results) == 1, f"Should have 1 pending fact, got {len(results)}"
    assert results[0]["person_name"] == "Hank", "Should show person display name"
    assert results[0]["fact_type"] == "BIRTH", "Should show fact type"
    assert results[0]["new_value"] == "1860", "Should show new value"
    assert results[0]["confidence_score"] == 80, "Should show confidence score"
    session.close()


def _test_batch_expire_old_items_db() -> None:
    """Test batch_expire_old_items expires items older than cutoff."""
    session = create_test_database()
    person = Person(username="ExpireTest", first_name="Iris")
    session.add(person)
    session.commit()

    old_date = datetime.now(UTC) - timedelta(days=45)
    recent_date = datetime.now(UTC) - timedelta(days=5)

    old_update = StagedUpdate(
        people_id=person.id,
        field_name="birth_year",
        proposed_value="1860",
        source="conversation",
        status=UpdateStatusEnum.PENDING,
        created_at=old_date,
    )
    recent_update = StagedUpdate(
        people_id=person.id,
        field_name="first_name",
        proposed_value="Iris May",
        source="conversation",
        status=UpdateStatusEnum.PENDING,
        created_at=recent_date,
    )
    old_conflict = DataConflict(
        people_id=person.id,
        field_name="first_name",
        existing_value="Iris",
        new_value="Iris May",
        source="gedcom_import",
        status=ConflictStatusEnum.OPEN,
        created_at=old_date,
    )
    session.add_all([old_update, recent_update, old_conflict])
    session.commit()

    queue = ReviewQueue()
    expired_updates, expired_conflicts = queue.batch_expire_old_items(session, max_age_days=30)
    session.commit()

    assert expired_updates == 1, f"Should expire 1 old update, got {expired_updates}"
    assert expired_conflicts == 1, f"Should expire 1 old conflict, got {expired_conflicts}"

    # Recent update should still be pending
    remaining = (
        session.query(StagedUpdate)
        .filter(StagedUpdate.status == UpdateStatusEnum.PENDING)
        .all()
    )
    assert len(remaining) == 1, "Recent update should still be PENDING"
    assert remaining[0].field_name == "first_name", "Recent update should be the first_name one"
    session.close()


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


# --- Test Suite Setup ---


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Review Queue Action", "actions/action_review.py")

    # Dataclass / formatting tests (kept - test real computation)
    suite.run_test("Review item display diff", _test_review_item_display_diff)
    suite.run_test("Review item display diff empty", _test_review_item_display_diff_empty)
    suite.run_test("Display item formatting", _test_display_item_format)
    suite.run_test("Display item conflict type", _test_display_item_conflict)
    suite.run_test("Format summary", _test_format_summary)

    # DB-backed staged update tests
    suite.run_test("Approve staged update (DB)", _test_approve_staged_update_db)
    suite.run_test("Approve staged update not found", _test_approve_staged_update_not_found)
    suite.run_test("Approve already-approved update", _test_approve_already_approved)
    suite.run_test("Reject staged update (DB)", _test_reject_staged_update_db)
    suite.run_test("Reject staged update not found", _test_reject_staged_update_not_found)

    # DB-backed conflict resolution tests
    suite.run_test("Resolve conflict accept (DB)", _test_resolve_conflict_accept)
    suite.run_test("Resolve conflict reject (DB)", _test_resolve_conflict_reject)

    # DB-backed pending items and summary tests
    suite.run_test("Get pending items (DB)", _test_get_pending_items_db)
    suite.run_test("Get pending items excludes non-pending", _test_get_pending_items_excludes_non_pending)
    suite.run_test("Get summary (DB)", _test_get_summary_db)

    # DB-backed SuggestedFact tests
    suite.run_test("Approve suggested fact (DB)", _test_approve_suggested_fact_db)
    suite.run_test("Approve suggested fact not found", _test_approve_suggested_fact_not_found)
    suite.run_test("Reject suggested fact (DB)", _test_reject_suggested_fact_db)
    suite.run_test("Reject suggested fact already approved", _test_reject_suggested_fact_already_approved)
    suite.run_test("List pending suggested facts (DB)", _test_list_pending_suggested_facts_db)

    # DB-backed batch expiry tests
    suite.run_test("Batch expire old items (DB)", _test_batch_expire_old_items_db)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
