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

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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


if __name__ == "__main__":
    sys.exit(main())
