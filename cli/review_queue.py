#!/usr/bin/env python3
"""
Review Queue CLI - Command-line interface for managing draft approvals.

Provides standalone CLI commands for the draft review workflow as documented
in docs/specs/operator_manual.md.

Usage:
    python -m cli.review_queue list [--limit N] [--priority PRIORITY]
    python -m cli.review_queue view --id ID
    python -m cli.review_queue approve --id ID
    python -m cli.review_queue reject --id ID --reason REASON
    python -m cli.review_queue stats
"""


import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging

logger = logging.getLogger(__name__)


def _get_db_session() -> Any:
    """Get database session for queue operations."""
    from core.database_manager import DatabaseManager

    db_manager = DatabaseManager()
    session = db_manager.get_session()
    if session is None:
        raise RuntimeError("Failed to create database session")
    return session


def _get_queue_service(session: Any) -> Any:
    """Get ApprovalQueueService instance."""
    from core.approval_queue import ApprovalQueueService

    return ApprovalQueueService(session)


def cmd_list(args: argparse.Namespace) -> int:
    """List pending drafts in the review queue."""
    session = _get_db_session()
    service = _get_queue_service(session)

    try:
        drafts = service.get_pending_queue(limit=args.limit)

        # Filter by priority if specified
        if args.priority:
            priority_upper = args.priority.upper()
            drafts = [d for d in drafts if getattr(d, "priority", "NORMAL").upper() == priority_upper]

        if not drafts:
            print("✅ No pending drafts in the review queue.")
            return 0

        # Print header
        print(f"\n📋 REVIEW QUEUE - {len(drafts)} Pending Draft(s)")
        print("=" * 80)
        print(f"{'ID':>6} │ {'Person':<20} │ {'Confidence':>10} │ {'Priority':<10} │ {'Created':<16}")
        print("-" * 80)

        for draft in drafts:
            draft_id = getattr(draft, "id", "?")
            person = getattr(draft, "person", None)
            person_name = (
                (getattr(person, "display_name", None) or getattr(person, "username", None) or "Unknown")
                if person
                else "Unknown"
            )
            confidence = getattr(draft, "ai_confidence", 0) or 0
            priority = getattr(draft, "priority", "NORMAL")
            created = getattr(draft, "created_at", None)
            created_str = created.strftime("%Y-%m-%d %H:%M") if created else "Unknown"

            print(f"{draft_id:>6} │ {person_name[:20]:<20} │ {confidence:>9}% │ {priority:<10} │ {created_str:<16}")

        print("=" * 80)
        return 0

    finally:
        session.close()


def cmd_view(args: argparse.Namespace) -> int:
    """View full details of a specific draft."""
    session = _get_db_session()
    service = _get_queue_service(session)

    try:
        draft = service.get_draft_by_id(args.id)
        if not draft:
            print(f"❌ Draft #{args.id} not found.")
            return 1

        person = getattr(draft, "person", None)
        person_name = (
            (getattr(person, "display_name", None) or getattr(person, "username", None) or "Unknown")
            if person
            else "Unknown"
        )

        print("\n" + "═" * 70)
        print(f"DRAFT #{args.id} - {person_name}")
        print("═" * 70)
        print(f"📅 Created: {getattr(draft, 'created_at', 'Unknown')}")
        print(f"📊 AI Confidence: {getattr(draft, 'ai_confidence', 0)}%")
        print(f"🎯 Priority: {getattr(draft, 'priority', 'NORMAL')}")
        print(f"📝 Status: {getattr(draft, 'status', 'UNKNOWN')}")
        print("-" * 70)
        print("CONTENT:")
        print("-" * 70)
        print(getattr(draft, "content", "(no content)"))
        print("═" * 70)

        return 0

    finally:
        session.close()


def cmd_approve(args: argparse.Namespace) -> int:
    """Approve a draft for sending."""
    session = _get_db_session()
    service = _get_queue_service(session)

    try:
        success = service.approve_draft(args.id)
        if success:
            print(f"✅ Draft #{args.id} approved successfully.")
            return 0
        print(f"❌ Failed to approve draft #{args.id}. It may not exist or may not be pending.")
        return 1

    finally:
        session.close()


def cmd_reject(args: argparse.Namespace) -> int:
    """Reject a draft with a reason."""
    session = _get_db_session()
    service = _get_queue_service(session)

    try:
        success = service.reject_draft(args.id, reason=args.reason)
        if success:
            print(f"✅ Draft #{args.id} rejected: {args.reason}")
            return 0
        print(f"❌ Failed to reject draft #{args.id}. It may not exist or may not be pending.")
        return 1

    finally:
        session.close()


def cmd_stats(_args: argparse.Namespace) -> int:
    """Display queue statistics."""
    session = _get_db_session()
    service = _get_queue_service(session)

    try:
        stats = service.get_queue_stats()

        print("\n📊 REVIEW QUEUE STATISTICS")
        print("=" * 40)
        print(f"  Pending:       {stats.pending_count:>6}")
        print(f"  Auto-Approved: {stats.auto_approved_count:>6}")
        print(f"  Approved Today:{stats.approved_today:>6}")
        print(f"  Rejected Today:{stats.rejected_today:>6}")
        print(f"  Expired:       {stats.expired_count:>6}")
        print("-" * 40)
        print(f"  Total Approved:{stats.total_approved:>6}")
        print(f"  Total Rejected:{stats.total_rejected:>6}")
        print(f"  Acceptance Rate: {stats.acceptance_rate:>5.1f}%")
        if stats.by_priority:
            print("-" * 40)
            print("  By Priority:")
            for priority, count in stats.by_priority.items():
                print(f"    {priority:<12} {count:>6}")
        print("=" * 40)

        return 0

    finally:
        session.close()


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m cli.review_queue",
        description="Review Queue CLI - Manage draft approvals for genealogical messaging",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List pending drafts")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum drafts to show (default: 50)")
    list_parser.add_argument("--priority", choices=["normal", "high", "critical"], help="Filter by priority")
    list_parser.set_defaults(func=cmd_list)

    # view command
    view_parser = subparsers.add_parser("view", help="View draft details")
    view_parser.add_argument("--id", type=int, required=True, help="Draft ID to view")
    view_parser.set_defaults(func=cmd_view)

    # approve command
    approve_parser = subparsers.add_parser("approve", help="Approve a draft")
    approve_parser.add_argument("--id", type=int, required=True, help="Draft ID to approve")
    approve_parser.set_defaults(func=cmd_approve)

    # reject command
    reject_parser = subparsers.add_parser("reject", help="Reject a draft")
    reject_parser.add_argument("--id", type=int, required=True, help="Draft ID to reject")
    reject_parser.add_argument("--reason", type=str, required=True, help="Rejection reason")
    reject_parser.set_defaults(func=cmd_reject)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="View queue statistics")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


def _cli_entry_point() -> None:
    """Entry point when run as script or module."""
    import os

    if os.environ.get("RUN_MODULE_TESTS") == "1":
        sys.exit(0 if run_comprehensive_tests() else 1)
    else:
        sys.exit(main())


# =============================================================================
# Tests
# =============================================================================


def module_tests() -> bool:
    """Embedded tests for review_queue CLI."""
    from datetime import datetime
    from io import StringIO
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from testing.test_framework import TestSuite

    suite = TestSuite("Review Queue CLI", "cli/review_queue.py")

    # Resolve actual module name (__main__ when run directly, cli.review_queue when imported)
    test_mod = __name__

    # ── Argument parsing ─────────────────────────────────────────

    def _test_main_no_args() -> bool:
        """main() with no command prints help and returns 0."""
        with patch("sys.argv", ["prog"]):
            result = main()
        return result == 0

    suite.run_test("main() no args returns 0", _test_main_no_args)

    def _test_parser_list_defaults() -> bool:
        """'list' subcommand parses with defaults."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        lp = subparsers.add_parser("list")
        lp.add_argument("--limit", type=int, default=50)
        lp.add_argument("--priority", choices=["normal", "high", "critical"])
        args = parser.parse_args(["list"])
        return args.command == "list" and args.limit == 50 and args.priority is None

    suite.run_test("parser list defaults", _test_parser_list_defaults)

    def _test_parser_list_with_options() -> bool:
        """'list --limit 10 --priority high' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        lp = subparsers.add_parser("list")
        lp.add_argument("--limit", type=int, default=50)
        lp.add_argument("--priority", choices=["normal", "high", "critical"])
        args = parser.parse_args(["list", "--limit", "10", "--priority", "high"])
        return args.limit == 10 and args.priority == "high"

    suite.run_test("parser list with options", _test_parser_list_with_options)

    # ── cmd_list ─────────────────────────────────────────────────

    def _test_cmd_list_empty() -> bool:
        """cmd_list returns 0 when no drafts."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        mock_service.get_pending_queue.return_value = []
        args = SimpleNamespace(limit=50, priority=None)

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service):
            result = cmd_list(args)

        return result == 0

    suite.run_test("cmd_list empty queue", _test_cmd_list_empty)

    def _test_cmd_list_with_drafts() -> bool:
        """cmd_list returns 0 and prints drafts."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        draft = SimpleNamespace(
            id=42,
            person=SimpleNamespace(display_name="Jane Doe", username="jdoe"),
            ai_confidence=85,
            priority="HIGH",
            created_at=datetime(2025, 1, 15, 10, 30),
        )
        mock_service.get_pending_queue.return_value = [draft]
        args = SimpleNamespace(limit=50, priority=None)

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service), \
             patch("sys.stdout", new_callable=StringIO) as out:
            result = cmd_list(args)

        return result == 0 and "Jane Doe" in out.getvalue() and "42" in out.getvalue()

    suite.run_test("cmd_list with drafts", _test_cmd_list_with_drafts)

    def _test_cmd_list_priority_filter() -> bool:
        """cmd_list filters by priority."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        d1 = SimpleNamespace(id=1, person=None, ai_confidence=80, priority="HIGH", created_at=None)
        d2 = SimpleNamespace(id=2, person=None, ai_confidence=60, priority="NORMAL", created_at=None)
        mock_service.get_pending_queue.return_value = [d1, d2]
        args = SimpleNamespace(limit=50, priority="high")

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service), \
             patch("sys.stdout", new_callable=StringIO) as out:
            result = cmd_list(args)

        output = out.getvalue()
        # Should show 1 draft (HIGH only), not the NORMAL one
        return result == 0 and "1 Pending" in output

    suite.run_test("cmd_list priority filter", _test_cmd_list_priority_filter)

    # ── cmd_view ─────────────────────────────────────────────────

    def _test_cmd_view_found() -> bool:
        """cmd_view returns 0 when draft exists."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        draft = SimpleNamespace(
            id=7,
            person=SimpleNamespace(display_name="Bob", username="bob"),
            created_at="2025-01-01",
            ai_confidence=90,
            priority="NORMAL",
            status="PENDING",
            content="Hello, I noticed we share DNA...",
        )
        mock_service.get_draft_by_id.return_value = draft
        args = SimpleNamespace(id=7)

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service), \
             patch("sys.stdout", new_callable=StringIO) as out:
            result = cmd_view(args)

        return result == 0 and "Bob" in out.getvalue() and "share DNA" in out.getvalue()

    suite.run_test("cmd_view found", _test_cmd_view_found)

    def _test_cmd_view_not_found() -> bool:
        """cmd_view returns 1 when draft missing."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        mock_service.get_draft_by_id.return_value = None
        args = SimpleNamespace(id=999)

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service):
            result = cmd_view(args)

        return result == 1

    suite.run_test("cmd_view not found", _test_cmd_view_not_found)

    # ── cmd_approve / cmd_reject ─────────────────────────────────

    def _test_cmd_approve_success() -> bool:
        """cmd_approve returns 0 on success."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        mock_service.approve_draft.return_value = True
        args = SimpleNamespace(id=10)

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service):
            result = cmd_approve(args)

        return result == 0

    suite.run_test("cmd_approve success", _test_cmd_approve_success)

    def _test_cmd_approve_failure() -> bool:
        """cmd_approve returns 1 on failure."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        mock_service.approve_draft.return_value = False
        args = SimpleNamespace(id=10)

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service):
            result = cmd_approve(args)

        return result == 1

    suite.run_test("cmd_approve failure", _test_cmd_approve_failure)

    def _test_cmd_reject_success() -> bool:
        """cmd_reject returns 0 on success."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        mock_service.reject_draft.return_value = True
        args = SimpleNamespace(id=10, reason="Not relevant")

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service):
            result = cmd_reject(args)

        return result == 0

    suite.run_test("cmd_reject success", _test_cmd_reject_success)

    # ── cmd_stats ────────────────────────────────────────────────

    def _test_cmd_stats() -> bool:
        """cmd_stats returns 0 and prints statistics."""
        mock_session = MagicMock()
        mock_service = MagicMock()
        stats = SimpleNamespace(
            pending_count=5,
            auto_approved_count=12,
            approved_today=3,
            rejected_today=1,
            expired_count=0,
            total_approved=100,
            total_rejected=20,
            acceptance_rate=83.3,
            by_priority={"HIGH": 2, "NORMAL": 3},
        )
        mock_service.get_queue_stats.return_value = stats
        args = SimpleNamespace()

        with patch(f"{test_mod}._get_db_session", return_value=mock_session), \
             patch(f"{test_mod}._get_queue_service", return_value=mock_service), \
             patch("sys.stdout", new_callable=StringIO) as out:
            result = cmd_stats(args)

        output = out.getvalue()
        return result == 0 and "83.3" in output and "HIGH" in output

    suite.run_test("cmd_stats display", _test_cmd_stats)

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Standard test runner entry point."""
    from testing.test_framework import create_standard_test_runner

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    _cli_entry_point()
