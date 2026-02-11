#!/usr/bin/env python3
"""
Facts Queue CLI - Command-line interface for managing extracted fact approvals.

Provides standalone CLI commands for the facts review workflow, allowing operators
to review, approve, or reject genealogical facts extracted from conversations.

Usage:
    python -m cli.facts_queue list [--limit N] [--status STATUS] [--type TYPE]
    python -m cli.facts_queue view --id ID
    python -m cli.facts_queue approve --id ID
    python -m cli.facts_queue reject --id ID --reason REASON
    python -m cli.facts_queue stats
    python -m cli.facts_queue conflicts [--limit N] [--severity SEVERITY]

Phase 3.3: Review Queue for Facts implementation.
"""


import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.database import (
    ConflictSeverityEnum,
    ConflictStatusEnum,
    DataConflict,
    FactStatusEnum,
    FactTypeEnum,
    Person,
    SuggestedFact,
)
from core.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class FactsQueueStats:
    """Statistics for the facts review queue."""

    pending_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    by_type: dict[str, int] | None = None
    avg_confidence: float = 0.0
    approval_rate: float = 0.0


@dataclass
class ConflictsStats:
    """Statistics for data conflicts."""

    open_count: int = 0
    resolved_count: int = 0
    rejected_count: int = 0
    by_severity: dict[str, int] | None = None


# ==============================================================================
# Database Utilities
# ==============================================================================


def _get_db_session() -> Session:
    """Get database session for queue operations."""
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    if session is None:
        raise RuntimeError("Failed to create database session")
    return session


# ==============================================================================
# Facts Queue Service
# ==============================================================================


class FactsQueueService:
    """Service for managing the facts review queue."""

    def __init__(self, session: Session) -> None:
        """Initialize with database session."""
        self.session = session

    def get_pending_facts(
        self,
        limit: int = 50,
        fact_type: FactTypeEnum | None = None,
    ) -> list[SuggestedFact]:
        """
        Get pending facts for review.

        Args:
            limit: Maximum number of facts to return.
            fact_type: Optional filter by fact type.

        Returns:
            List of pending SuggestedFact records.
        """
        query = self.session.query(SuggestedFact).filter(SuggestedFact.status == FactStatusEnum.PENDING)

        if fact_type:
            query = query.filter(SuggestedFact.fact_type == fact_type)

        return query.order_by(SuggestedFact.created_at.desc()).limit(limit).all()

    def get_facts_by_status(
        self,
        status: FactStatusEnum,
        limit: int = 50,
    ) -> list[SuggestedFact]:
        """Get facts filtered by status."""
        return (
            self.session.query(SuggestedFact)
            .filter(SuggestedFact.status == status)
            .order_by(SuggestedFact.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_fact_by_id(self, fact_id: int) -> SuggestedFact | None:
        """Get a specific fact by ID."""
        return self.session.query(SuggestedFact).filter(SuggestedFact.id == fact_id).first()

    def approve_fact(self, fact_id: int) -> bool:
        """
        Approve a pending fact.

        Args:
            fact_id: ID of the fact to approve.

        Returns:
            True if successful, False otherwise.
        """
        fact = self.get_fact_by_id(fact_id)
        if not fact or fact.status != FactStatusEnum.PENDING:
            return False

        fact.status = FactStatusEnum.APPROVED
        fact.updated_at = datetime.now(UTC)
        self.session.commit()
        logger.info(f"Approved fact #{fact_id}")
        return True

    def reject_fact(self, fact_id: int, reason: str | None = None) -> bool:
        """
        Reject a pending fact.

        Args:
            fact_id: ID of the fact to reject.
            reason: Optional rejection reason (logged but not stored on model).

        Returns:
            True if successful, False otherwise.
        """
        fact = self.get_fact_by_id(fact_id)
        if not fact or fact.status != FactStatusEnum.PENDING:
            return False

        fact.status = FactStatusEnum.REJECTED
        fact.updated_at = datetime.now(UTC)
        self.session.commit()
        logger.info(f"Rejected fact #{fact_id}: {reason or 'No reason provided'}")
        return True

    def get_queue_stats(self) -> FactsQueueStats:
        """Get statistics for the facts queue."""
        pending = self.session.query(SuggestedFact).filter(SuggestedFact.status == FactStatusEnum.PENDING).count()
        approved = self.session.query(SuggestedFact).filter(SuggestedFact.status == FactStatusEnum.APPROVED).count()
        rejected = self.session.query(SuggestedFact).filter(SuggestedFact.status == FactStatusEnum.REJECTED).count()

        # By type breakdown
        by_type_query = (
            self.session.query(SuggestedFact.fact_type, func.count(SuggestedFact.id))
            .filter(SuggestedFact.status == FactStatusEnum.PENDING)
            .group_by(SuggestedFact.fact_type)
            .all()
        )
        by_type = {ft.value: count for ft, count in by_type_query}

        # Average confidence for pending
        avg_conf_result = (
            self.session.query(func.avg(SuggestedFact.confidence_score))
            .filter(SuggestedFact.status == FactStatusEnum.PENDING)
            .scalar()
        )
        avg_confidence = float(avg_conf_result) if avg_conf_result else 0.0

        # Approval rate
        total_reviewed = approved + rejected
        approval_rate = (approved / total_reviewed * 100) if total_reviewed > 0 else 0.0

        return FactsQueueStats(
            pending_count=pending,
            approved_count=approved,
            rejected_count=rejected,
            by_type=by_type if by_type else None,
            avg_confidence=avg_confidence,
            approval_rate=approval_rate,
        )

    def get_open_conflicts(
        self,
        limit: int = 50,
        severity: ConflictSeverityEnum | None = None,
    ) -> list[DataConflict]:
        """
        Get open data conflicts for review.

        Args:
            limit: Maximum number of conflicts to return.
            severity: Optional filter by severity level.

        Returns:
            List of open DataConflict records.
        """
        query = self.session.query(DataConflict).filter(DataConflict.status == ConflictStatusEnum.OPEN)

        if severity:
            query = query.filter(DataConflict.severity == severity)

        # Order by created_at desc (SQLAlchemy doesn't support direct enum ordering)
        return query.order_by(DataConflict.created_at.desc()).limit(limit).all()

    def get_conflict_stats(self) -> ConflictsStats:
        """Get statistics for data conflicts."""
        open_count = self.session.query(DataConflict).filter(DataConflict.status == ConflictStatusEnum.OPEN).count()
        resolved = self.session.query(DataConflict).filter(DataConflict.status == ConflictStatusEnum.RESOLVED).count()
        rejected = self.session.query(DataConflict).filter(DataConflict.status == ConflictStatusEnum.REJECTED).count()

        # By severity breakdown
        by_sev_query = (
            self.session.query(DataConflict.severity, func.count(DataConflict.id))
            .filter(DataConflict.status == ConflictStatusEnum.OPEN)
            .group_by(DataConflict.severity)
            .all()
        )
        by_severity = {sev.value: count for sev, count in by_sev_query}

        return ConflictsStats(
            open_count=open_count,
            resolved_count=resolved,
            rejected_count=rejected,
            by_severity=by_severity if by_severity else None,
        )

    def create_tasks_for_critical_conflicts(
        self,
        severity_filter: list[ConflictSeverityEnum] | None = None,
    ) -> tuple[int, int]:
        """
        Create MS To-Do tasks for HIGH and CRITICAL severity conflicts.

        Phase 3.4: MS To-Do Integration for MAJOR_CONFLICT items.

        Args:
            severity_filter: List of severities to create tasks for.
                            Defaults to [CRITICAL, HIGH].

        Returns:
            Tuple of (tasks_created, tasks_failed).
        """
        from integrations import ms_graph_utils

        # Default to critical and high severity
        if severity_filter is None:
            severity_filter = [ConflictSeverityEnum.CRITICAL, ConflictSeverityEnum.HIGH]

        # Get conflicts that need tasks
        conflicts = (
            self.session.query(DataConflict)
            .filter(
                DataConflict.status == ConflictStatusEnum.OPEN,
                DataConflict.severity.in_(severity_filter),
            )
            .all()
        )

        if not conflicts:
            logger.info("No critical/high severity conflicts found for task creation.")
            return (0, 0)

        # Acquire MS Graph token
        token = ms_graph_utils.acquire_token_device_flow()
        if not token:
            logger.warning("MS Graph authentication failed. Cannot create tasks.")
            return (0, len(conflicts))

        # Get or create todo list
        list_name = "Ancestry Research"
        list_id = ms_graph_utils.get_todo_list_id(token, list_name)
        if not list_id:
            logger.warning(f"MS To-Do list '{list_name}' not found. Cannot create tasks.")
            return (0, len(conflicts))

        tasks_created = 0
        tasks_failed = 0

        for conflict in conflicts:
            person = conflict.person
            person_name = _get_person_display_name(person)

            # Build task title
            severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠"}.get(
                conflict.severity.value if conflict.severity else "HIGH", "⚠️"
            )
            task_title = f"{severity_emoji} Conflict: {person_name} - {conflict.field_name}"

            # Build task body with conflict details
            task_body = _format_conflict_task_body(conflict, person_name)

            # Determine importance based on severity
            importance = "high" if conflict.severity == ConflictSeverityEnum.CRITICAL else "normal"

            # Create categories for organization
            categories = [
                f"fact_type:{conflict.field_name or 'unknown'}",
                f"person:{person_name[:30]}",
                f"severity:{conflict.severity.value if conflict.severity else 'unknown'}",
            ]

            # Create the task
            task_id = ms_graph_utils.create_todo_task(
                access_token=token,
                list_id=list_id,
                task_title=task_title[:255],  # MS Graph title limit
                task_body=task_body,
                importance=importance,
                categories=categories,
            )

            if task_id:
                tasks_created += 1
                logger.info(f"Created MS To-Do task for conflict #{conflict.id}: {task_title[:50]}...")
            else:
                tasks_failed += 1
                logger.warning(f"Failed to create task for conflict #{conflict.id}")

        return (tasks_created, tasks_failed)


def _format_conflict_task_body(conflict: DataConflict, person_name: str) -> str:
    """Format the task body with conflict details."""
    lines = [
        f"📋 Data Conflict for {person_name}",
        "",
        f"Field: {conflict.field_name}",
        f"Existing Value: {conflict.existing_value or '(empty)'}",
        f"New Value: {conflict.new_value or '(empty)'}",
        f"Severity: {conflict.severity.value if conflict.severity else 'UNKNOWN'}",
        f"Source: {conflict.source or 'conversation'}",
        "",
        "Action Required:",
        "- Research to verify which value is correct",
        "- Update the tree if new value is confirmed",
        "- Resolve the conflict in facts_queue CLI",
        "",
        f"Conflict ID: #{conflict.id}",
        f"Created: {conflict.created_at}",
    ]

    if conflict.confidence_score:
        lines.insert(6, f"AI Confidence: {conflict.confidence_score}%")

    return "\n".join(lines)


# ==============================================================================
# CLI Commands
# ==============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """List pending facts in the review queue."""
    session = _get_db_session()
    service = FactsQueueService(session)

    try:
        # Parse fact type filter
        fact_type = None
        if args.type:
            try:
                fact_type = FactTypeEnum(args.type.upper())
            except ValueError:
                print(f"❌ Invalid fact type: {args.type}")
                return 1

        # Get facts based on status filter
        if args.status:
            try:
                status_enum = FactStatusEnum(args.status.upper())
                facts = service.get_facts_by_status(status_enum, limit=args.limit)
            except ValueError:
                print(f"❌ Invalid status: {args.status}")
                return 1
        else:
            facts = service.get_pending_facts(limit=args.limit, fact_type=fact_type)

        if not facts:
            status_label = args.status.upper() if args.status else "PENDING"
            print(f"✅ No {status_label} facts in the review queue.")
            return 0

        # Print header
        print(f"\n📋 FACTS QUEUE - {len(facts)} Fact(s)")
        print("=" * 100)
        print(f"{'ID':>6} │ {'Person':<18} │ {'Type':<12} │ {'Conf':>4} │ {'Status':<10} │ {'Value':<30}")
        print("-" * 100)

        for fact in facts:
            fact_id = fact.id
            person = fact.person
            person_name = _get_person_display_name(person)[:18]
            fact_type_str = fact.fact_type.value if fact.fact_type else "OTHER"
            confidence = fact.confidence_score or 0
            status = fact.status.value if fact.status else "UNKNOWN"
            value = (fact.new_value or "")[:30]

            print(
                f"{fact_id:>6} │ {person_name:<18} │ {fact_type_str:<12} │ {confidence:>3}% │ {status:<10} │ {value:<30}"
            )

        print("=" * 100)
        return 0

    finally:
        session.close()


def cmd_view(args: argparse.Namespace) -> int:
    """View full details of a specific fact."""
    session = _get_db_session()
    service = FactsQueueService(session)

    try:
        fact = service.get_fact_by_id(args.id)
        if not fact:
            print(f"❌ Fact #{args.id} not found.")
            return 1

        person = fact.person
        person_name = _get_person_display_name(person)

        print("\n" + "═" * 70)
        print(f"FACT #{args.id} - {person_name}")
        print("═" * 70)
        print(f"📅 Created:    {fact.created_at}")
        print(f"📊 Type:       {fact.fact_type.value if fact.fact_type else 'OTHER'}")
        print(f"🎯 Confidence: {fact.confidence_score or 0}%")
        print(f"📝 Status:     {fact.status.value if fact.status else 'UNKNOWN'}")
        print(f"📄 Source ID:  {fact.source_message_id or 'N/A'}")
        print("-" * 70)
        print("ORIGINAL VALUE:")
        print(fact.original_value or "(none)")
        print("-" * 70)
        print("EXTRACTED VALUE:")
        print(fact.new_value or "(none)")
        print("═" * 70)

        return 0

    finally:
        session.close()


def cmd_approve(args: argparse.Namespace) -> int:
    """Approve a fact."""
    session = _get_db_session()
    service = FactsQueueService(session)

    try:
        success = service.approve_fact(args.id)
        if success:
            print(f"✅ Fact #{args.id} approved successfully.")
            return 0
        print(f"❌ Failed to approve fact #{args.id}. It may not exist or may not be pending.")
        return 1

    finally:
        session.close()


def cmd_reject(args: argparse.Namespace) -> int:
    """Reject a fact with a reason."""
    session = _get_db_session()
    service = FactsQueueService(session)

    try:
        success = service.reject_fact(args.id, reason=args.reason)
        if success:
            print(f"✅ Fact #{args.id} rejected: {args.reason}")
            return 0
        print(f"❌ Failed to reject fact #{args.id}. It may not exist or may not be pending.")
        return 1

    finally:
        session.close()


def cmd_stats(_args: argparse.Namespace) -> int:
    """Display queue statistics."""
    session = _get_db_session()
    service = FactsQueueService(session)

    try:
        stats = service.get_queue_stats()

        print("\n📊 FACTS QUEUE STATISTICS")
        print("=" * 40)
        print(f"  Pending:        {stats.pending_count:>6}")
        print(f"  Approved:       {stats.approved_count:>6}")
        print(f"  Rejected:       {stats.rejected_count:>6}")
        print(f"  Avg Confidence: {stats.avg_confidence:>5.1f}%")
        print(f"  Approval Rate:  {stats.approval_rate:>5.1f}%")
        if stats.by_type:
            print("-" * 40)
            print("  Pending by Type:")
            for fact_type, count in sorted(stats.by_type.items()):
                print(f"    {fact_type:<12} {count:>6}")
        print("=" * 40)

        return 0

    finally:
        session.close()


def cmd_conflicts(args: argparse.Namespace) -> int:
    """List open data conflicts."""
    session = _get_db_session()
    service = FactsQueueService(session)

    try:
        # Parse severity filter
        severity = None
        if args.severity:
            try:
                severity = ConflictSeverityEnum(args.severity.upper())
            except ValueError:
                print(f"❌ Invalid severity: {args.severity}")
                return 1

        conflicts = service.get_open_conflicts(limit=args.limit, severity=severity)

        if not conflicts:
            print("✅ No open data conflicts.")
            return 0

        # Print header
        print(f"\n⚠️ DATA CONFLICTS - {len(conflicts)} Open Conflict(s)")
        print("=" * 110)
        print(f"{'ID':>6} │ {'Person':<18} │ {'Field':<15} │ {'Severity':<10} │ {'Existing':<18} │ {'New':<18}")
        print("-" * 110)

        for conflict in conflicts:
            conflict_id = conflict.id
            person = conflict.person
            person_name = _get_person_display_name(person)[:18]
            field_name = (conflict.field_name or "unknown")[:15]
            severity_str = conflict.severity.value if conflict.severity else "MEDIUM"
            existing = (conflict.existing_value or "")[:18]
            new = (conflict.new_value or "")[:18]

            # Color-code severity with emoji
            sev_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity_str, "⚪")

            print(
                f"{conflict_id:>6} │ {person_name:<18} │ {field_name:<15} │ "
                f"{sev_emoji}{severity_str:<9} │ {existing:<18} │ {new:<18}"
            )

        print("=" * 110)

        # Also show conflict stats
        conflict_stats = service.get_conflict_stats()
        if conflict_stats.by_severity:
            print("\n📊 By Severity:")
            for sev, count in sorted(conflict_stats.by_severity.items()):
                print(f"  {sev}: {count}")

        return 0

    finally:
        session.close()


def cmd_create_tasks(args: argparse.Namespace) -> int:
    """Create MS To-Do tasks for critical conflicts."""
    session = _get_db_session()
    service = FactsQueueService(session)

    try:
        # Determine severity filter
        severity_filter = None
        if args.severity:
            try:
                sev = ConflictSeverityEnum(args.severity.upper())
                severity_filter = [sev]
            except ValueError:
                print(f"❌ Invalid severity: {args.severity}")
                return 1

        print("📝 Creating MS To-Do tasks for critical conflicts...")
        tasks_created, tasks_failed = service.create_tasks_for_critical_conflicts(severity_filter)

        if tasks_created == 0 and tasks_failed == 0:
            print("✅ No critical/high severity conflicts found requiring tasks.")
            return 0

        print("\n📊 Task Creation Summary:")
        print(f"  ✅ Created: {tasks_created}")
        print(f"  ❌ Failed:  {tasks_failed}")

        return 0 if tasks_failed == 0 else 1

    finally:
        session.close()


def _get_person_display_name(person: Person | None) -> str:
    """Get display name for a person."""
    if not person:
        return "Unknown"
    return getattr(person, "display_name", None) or getattr(person, "username", None) or f"Person #{person.id}"


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m cli.facts_queue",
        description="Facts Queue CLI - Manage extracted fact approvals for genealogical research",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List facts in the queue")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum facts to show (default: 50)")
    list_parser.add_argument(
        "--status",
        choices=["pending", "approved", "rejected"],
        help="Filter by status (default: pending)",
    )
    list_parser.add_argument(
        "--type",
        choices=["birth", "death", "relationship", "marriage", "location", "other"],
        help="Filter by fact type",
    )
    list_parser.set_defaults(func=cmd_list)

    # view command
    view_parser = subparsers.add_parser("view", help="View fact details")
    view_parser.add_argument("--id", type=int, required=True, help="Fact ID to view")
    view_parser.set_defaults(func=cmd_view)

    # approve command
    approve_parser = subparsers.add_parser("approve", help="Approve a fact")
    approve_parser.add_argument("--id", type=int, required=True, help="Fact ID to approve")
    approve_parser.set_defaults(func=cmd_approve)

    # reject command
    reject_parser = subparsers.add_parser("reject", help="Reject a fact")
    reject_parser.add_argument("--id", type=int, required=True, help="Fact ID to reject")
    reject_parser.add_argument("--reason", type=str, required=True, help="Rejection reason")
    reject_parser.set_defaults(func=cmd_reject)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="View queue statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # conflicts command
    conflicts_parser = subparsers.add_parser("conflicts", help="List open data conflicts")
    conflicts_parser.add_argument("--limit", type=int, default=50, help="Maximum conflicts to show (default: 50)")
    conflicts_parser.add_argument(
        "--severity",
        choices=["low", "medium", "high", "critical"],
        help="Filter by severity",
    )
    conflicts_parser.set_defaults(func=cmd_conflicts)

    # create-tasks command (Phase 3.4)
    create_tasks_parser = subparsers.add_parser(
        "create-tasks",
        help="Create MS To-Do tasks for critical conflicts",
    )
    create_tasks_parser.add_argument(
        "--severity",
        choices=["high", "critical"],
        help="Create tasks only for specific severity (default: both high and critical)",
    )
    create_tasks_parser.set_defaults(func=cmd_create_tasks)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


# ==============================================================================
# Test Suite
# ==============================================================================


def facts_queue_module_tests() -> bool:
    """Run tests for the facts queue module."""
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Facts Queue CLI", "cli/facts_queue.py")
        suite.start_suite()

        # Test FactsQueueStats dataclass
        def test_facts_queue_stats_defaults():
            stats = FactsQueueStats()
            assert stats.pending_count == 0
            assert stats.approved_count == 0
            assert stats.rejected_count == 0
            assert stats.by_type is None
            assert stats.avg_confidence == 0.0
            assert stats.approval_rate == 0.0

        def test_facts_queue_stats_with_values():
            stats = FactsQueueStats(
                pending_count=10,
                approved_count=50,
                rejected_count=5,
                by_type={"BIRTH": 5, "DEATH": 3},
                avg_confidence=75.5,
                approval_rate=90.9,
            )
            assert stats.pending_count == 10
            assert stats.approval_rate == 90.9
            assert stats.by_type is not None
            assert stats.by_type.get("BIRTH") == 5

        # Test ConflictsStats dataclass
        def test_conflicts_stats_defaults():
            stats = ConflictsStats()
            assert stats.open_count == 0
            assert stats.resolved_count == 0
            assert stats.rejected_count == 0
            assert stats.by_severity is None

        def test_conflicts_stats_with_values():
            stats = ConflictsStats(
                open_count=15,
                resolved_count=100,
                rejected_count=10,
                by_severity={"HIGH": 5, "CRITICAL": 2},
            )
            assert stats.open_count == 15
            assert stats.by_severity is not None
            assert stats.by_severity.get("CRITICAL") == 2

        # Test _get_person_display_name helper
        def test_get_person_display_name_none():
            result = _get_person_display_name(None)
            assert result == "Unknown"

        def test_get_person_display_name_with_display_name():
            from types import SimpleNamespace

            person = SimpleNamespace(display_name="John Doe", username="johndoe", id=123)
            result = _get_person_display_name(person)
            assert result == "John Doe"

        def test_get_person_display_name_fallback_to_username():
            from types import SimpleNamespace

            person = SimpleNamespace(display_name=None, username="johndoe", id=123)
            result = _get_person_display_name(person)
            assert result == "johndoe"

        # Test _format_conflict_task_body helper (Phase 3.4)
        def test_format_conflict_task_body():
            from types import SimpleNamespace

            conflict = SimpleNamespace(
                id=42, field_name="birth_year", existing_value="1850",
                new_value="1855", severity=ConflictSeverityEnum.HIGH,
                source="conversation", confidence_score=85,
                created_at="2025-12-15 10:30:00",
            )

            result = _format_conflict_task_body(conflict, "John Doe")
            assert "John Doe" in result
            assert "birth_year" in result
            assert "1850" in result
            assert "1855" in result
            assert "HIGH" in result
            assert "85%" in result
            assert "#42" in result

        suite.run_test(
            "FactsQueueStats defaults",
            test_facts_queue_stats_defaults,
            "Default values should be zero/None",
        )
        suite.run_test(
            "FactsQueueStats with values",
            test_facts_queue_stats_with_values,
            "Should store provided values correctly",
        )
        suite.run_test(
            "ConflictsStats defaults",
            test_conflicts_stats_defaults,
            "Default values should be zero/None",
        )
        suite.run_test(
            "ConflictsStats with values",
            test_conflicts_stats_with_values,
            "Should store provided values correctly",
        )
        suite.run_test(
            "Person display name None",
            test_get_person_display_name_none,
            "Should return 'Unknown' for None",
        )
        suite.run_test(
            "Person display name with display_name",
            test_get_person_display_name_with_display_name,
            "Should prefer display_name",
        )
        suite.run_test(
            "Person display name fallback",
            test_get_person_display_name_fallback_to_username,
            "Should fall back to username",
        )
        suite.run_test(
            "Format conflict task body",
            test_format_conflict_task_body,
            "Should format conflict details for MS To-Do task",
        )

        # ── FactsQueueService tests ──────────────────────────────

        def test_service_get_pending_facts():
            from unittest.mock import MagicMock
            mock_session = MagicMock()
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = []
            svc = FactsQueueService(mock_session)
            result = svc.get_pending_facts(limit=10)
            assert result == []

        suite.run_test("Service get_pending_facts empty", test_service_get_pending_facts)

        def test_service_get_facts_by_status():
            from unittest.mock import MagicMock
            mock_session = MagicMock()
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = ["fact1"]
            svc = FactsQueueService(mock_session)
            result = svc.get_facts_by_status(FactStatusEnum.APPROVED, limit=5)
            assert len(result) == 1

        suite.run_test("Service get_facts_by_status", test_service_get_facts_by_status)

        def test_service_get_fact_by_id():
            from unittest.mock import MagicMock
            mock_session = MagicMock()
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_fact = MagicMock()
            mock_fact.id = 42
            mock_query.first.return_value = mock_fact
            svc = FactsQueueService(mock_session)
            result = svc.get_fact_by_id(42)
            assert result is not None and result.id == 42

        suite.run_test("Service get_fact_by_id found", test_service_get_fact_by_id)

        def test_service_get_fact_by_id_not_found():
            from unittest.mock import MagicMock
            mock_session = MagicMock()
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = None
            svc = FactsQueueService(mock_session)
            result = svc.get_fact_by_id(999)
            assert result is None

        suite.run_test("Service get_fact_by_id not found", test_service_get_fact_by_id_not_found)

        def test_service_approve_fact():
            from unittest.mock import MagicMock, patch as _patch
            mock_session = MagicMock()
            mock_fact = MagicMock()
            mock_fact.status = FactStatusEnum.PENDING
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_fact
            svc = FactsQueueService(mock_session)
            result = svc.approve_fact(1)
            assert result is True
            assert mock_fact.status == FactStatusEnum.APPROVED
            mock_session.commit.assert_called_once()

        suite.run_test("Service approve_fact", test_service_approve_fact)

        def test_service_reject_fact():
            from unittest.mock import MagicMock
            mock_session = MagicMock()
            mock_fact = MagicMock()
            mock_fact.status = FactStatusEnum.PENDING
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_fact
            svc = FactsQueueService(mock_session)
            result = svc.reject_fact(1, reason="Incorrect")
            assert result is True
            assert mock_fact.status == FactStatusEnum.REJECTED
            mock_session.commit.assert_called_once()

        suite.run_test("Service reject_fact", test_service_reject_fact)

        def test_service_reject_not_pending():
            from unittest.mock import MagicMock
            mock_session = MagicMock()
            mock_fact = MagicMock()
            mock_fact.status = FactStatusEnum.APPROVED  # Already approved
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_fact
            svc = FactsQueueService(mock_session)
            result = svc.reject_fact(1)
            assert result is False

        suite.run_test("Service reject non-pending fails", test_service_reject_not_pending)

        # ── CLI command tests ────────────────────────────────────
        # Use __name__ for patch targets so patches work both when
        # imported as "cli.facts_queue" and when run as "__main__".
        mod = __name__

        def test_cmd_list_empty():
            from types import SimpleNamespace
            from unittest.mock import MagicMock, patch as _patch
            mock_session = MagicMock()
            mock_svc = MagicMock()
            mock_svc.get_pending_facts.return_value = []
            args = SimpleNamespace(limit=50, type=None, status=None)
            with _patch(f"{mod}._get_db_session", return_value=mock_session), \
                 _patch(f"{mod}.FactsQueueService", return_value=mock_svc):
                result = cmd_list(args)
            assert result == 0

        suite.run_test("cmd_list empty queue", test_cmd_list_empty)

        def test_cmd_view_not_found():
            from types import SimpleNamespace
            from unittest.mock import MagicMock, patch as _patch
            mock_session = MagicMock()
            mock_svc = MagicMock()
            mock_svc.get_fact_by_id.return_value = None
            args = SimpleNamespace(id=999)
            with _patch(f"{mod}._get_db_session", return_value=mock_session), \
                 _patch(f"{mod}.FactsQueueService", return_value=mock_svc):
                result = cmd_view(args)
            assert result == 1

        suite.run_test("cmd_view not found", test_cmd_view_not_found)

        def test_cmd_approve_success():
            from types import SimpleNamespace
            from unittest.mock import MagicMock, patch as _patch
            mock_session = MagicMock()
            mock_svc = MagicMock()
            mock_svc.approve_fact.return_value = True
            args = SimpleNamespace(id=1)
            with _patch(f"{mod}._get_db_session", return_value=mock_session), \
                 _patch(f"{mod}.FactsQueueService", return_value=mock_svc):
                result = cmd_approve(args)
            assert result == 0

        suite.run_test("cmd_approve success", test_cmd_approve_success)

        def test_cmd_reject_success():
            from types import SimpleNamespace
            from unittest.mock import MagicMock, patch as _patch
            mock_session = MagicMock()
            mock_svc = MagicMock()
            mock_svc.reject_fact.return_value = True
            args = SimpleNamespace(id=1, reason="Wrong data")
            with _patch(f"{mod}._get_db_session", return_value=mock_session), \
                 _patch(f"{mod}.FactsQueueService", return_value=mock_svc):
                result = cmd_reject(args)
            assert result == 0

        suite.run_test("cmd_reject success", test_cmd_reject_success)

        def test_cmd_stats_returns_0():
            from types import SimpleNamespace
            from unittest.mock import MagicMock, patch as _patch
            mock_session = MagicMock()
            mock_svc = MagicMock()
            mock_svc.get_queue_stats.return_value = FactsQueueStats(
                pending_count=5, approved_count=10, rejected_count=2,
                avg_confidence=80.0, approval_rate=83.3,
            )
            args = SimpleNamespace()
            with _patch(f"{mod}._get_db_session", return_value=mock_session), \
                 _patch(f"{mod}.FactsQueueService", return_value=mock_svc):
                result = cmd_stats(args)
            assert result == 0

        suite.run_test("cmd_stats returns 0", test_cmd_stats_returns_0)

        def test_cmd_conflicts_empty():
            from types import SimpleNamespace
            from unittest.mock import MagicMock, patch as _patch
            mock_session = MagicMock()
            mock_svc = MagicMock()
            mock_svc.get_open_conflicts.return_value = []
            args = SimpleNamespace(limit=50, severity=None)
            with _patch(f"{mod}._get_db_session", return_value=mock_session), \
                 _patch(f"{mod}.FactsQueueService", return_value=mock_svc):
                result = cmd_conflicts(args)
            assert result == 0

        suite.run_test("cmd_conflicts empty", test_cmd_conflicts_empty)

        def test_main_no_args():
            from unittest.mock import patch as _patch
            with _patch("sys.argv", ["prog"]):
                result = main()
            assert result == 0

        suite.run_test("main() no args returns 0", test_main_no_args)

        return suite.finish_suite()


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(facts_queue_module_tests)


if __name__ == "__main__":
    import os

    if os.environ.get("RUN_MODULE_TESTS") == "1":
        sys.exit(0 if run_comprehensive_tests() else 1)
    else:
        sys.exit(main())
