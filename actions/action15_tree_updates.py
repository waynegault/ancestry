"""Action 15: Apply Approved Tree Updates.

Provides a menu-driven interface for reviewing and applying APPROVED
SuggestedFacts to the Ancestry.com family tree via the TreeUpdateService.

This action:
- Lists pending SuggestedFacts awaiting approval
- Allows batch or individual approval/rejection
- Applies approved facts to the Ancestry tree via internal APIs
- Logs all operations to TreeUpdateLog for audit
- Supports dry-run mode for testing

Features:
- Interactive review queue with visual diffs
- Batch processing with rate limiting
- Rollback support via original value storage
- Integration with ReviewQueue workflow
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from sqlalchemy.orm import Session

from actions.action_review import ReviewQueue
from api.tree_update import (
    TreeOperationType,
    TreeUpdateResponse,
    TreeUpdateResult,
    TreeUpdateService,
)
from config import config_schema
from core.database import (
    FactStatusEnum,
    FactTypeEnum,
    Person,
    SuggestedFact,
    TreeUpdateLog,
    TreeUpdateStatusEnum,
)
from core.feature_flags import is_feature_enabled
from core.session_manager import SessionManager
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================


class TreeUpdateMode(Enum):
    """Modes for tree update operations."""

    DRY_RUN = "dry_run"  # Preview changes without applying
    APPLY = "apply"  # Apply changes to tree
    INTERACTIVE = "interactive"  # Prompt for each item


@dataclass
class TreeUpdateSummary:
    """Summary statistics for tree update operations."""

    total_pending: int = 0
    processed: int = 0
    applied: int = 0
    skipped: int = 0
    failed: int = 0
    by_fact_type: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# =============================================================================
# Main Action Functions
# =============================================================================


def run_tree_updates(
    session_manager: SessionManager,
    tree_id: str,
    mode: TreeUpdateMode = TreeUpdateMode.DRY_RUN,
    limit: int = 20,
    fact_types: Optional[list[FactTypeEnum]] = None,
) -> TreeUpdateSummary:
    """
    Process pending SuggestedFacts and apply to tree.

    Args:
        session_manager: Active SessionManager with authenticated session
        tree_id: Ancestry tree ID to update
        mode: Processing mode (DRY_RUN, APPLY, INTERACTIVE)
        limit: Maximum number of facts to process
        fact_types: Optional filter by fact type

    Returns:
        TreeUpdateSummary with processing statistics
    """
    start_time = time.time()
    summary = TreeUpdateSummary()

    # Check feature flag
    if not is_feature_enabled("ACTION15_TREE_UPDATES"):
        logger.warning("ACTION15_TREE_UPDATES feature flag is disabled")
        summary.errors.append("Feature flag disabled")
        return summary

    logger.info(f"🌳 Starting tree updates (mode={mode.value}, limit={limit})")

    with session_manager.db_transaction() as db_session:
        # Query pending suggested facts
        query = db_session.query(SuggestedFact).filter(SuggestedFact.status == FactStatusEnum.PENDING)

        if fact_types:
            query = query.filter(SuggestedFact.fact_type.in_(fact_types))

        pending_facts = query.order_by(SuggestedFact.created_at.asc()).limit(limit).all()

        summary.total_pending = len(pending_facts)
        logger.info(f"📋 Found {summary.total_pending} pending suggested facts")

        if summary.total_pending == 0:
            summary.duration_seconds = time.time() - start_time
            return summary

        # Process each fact
        tree_service = TreeUpdateService(session_manager)

        for fact in pending_facts:
            try:
                result = _process_single_fact(
                    db_session=db_session,
                    tree_service=tree_service,
                    fact=fact,
                    tree_id=tree_id,
                    mode=mode,
                )

                summary.processed += 1

                # Track by fact type
                fact_type_key = fact.fact_type.value if fact.fact_type else "unknown"
                summary.by_fact_type[fact_type_key] = summary.by_fact_type.get(fact_type_key, 0) + 1

                if result.result == TreeUpdateResult.SUCCESS:
                    summary.applied += 1
                    logger.info(f"✅ Applied fact {fact.id}: {result.message}")
                elif result.result == TreeUpdateResult.ALREADY_APPLIED:
                    summary.skipped += 1
                    logger.info(f"⏭️ Skipped fact {fact.id}: already applied")
                else:
                    summary.failed += 1
                    error_msg = f"Fact {fact.id}: {result.message}"
                    summary.errors.append(error_msg)
                    logger.warning(f"⚠️ Failed: {error_msg}")

            except Exception as e:
                summary.processed += 1
                summary.failed += 1
                error_msg = f"Fact {fact.id}: {e}"
                summary.errors.append(error_msg)
                logger.exception(f"Error processing fact {fact.id}")

    summary.duration_seconds = time.time() - start_time
    _log_summary(summary)
    return summary


def _process_single_fact(
    db_session: Session,
    tree_service: TreeUpdateService,
    fact: SuggestedFact,
    tree_id: str,
    mode: TreeUpdateMode,
) -> TreeUpdateResponse:
    """
    Process a single suggested fact.

    Args:
        db_session: Database session
        tree_service: TreeUpdateService instance
        fact: SuggestedFact to process
        tree_id: Target tree ID
        mode: Processing mode

    Returns:
        TreeUpdateResponse with result details
    """
    # Get person for display
    person = db_session.query(Person).filter(Person.id == fact.people_id).first()
    person_name = person.display_name if person else f"ID:{fact.people_id}"

    logger.debug(
        f"Processing fact {fact.id}: {fact.fact_type.value if fact.fact_type else 'unknown'} for {person_name}"
    )

    # In DRY_RUN mode, just preview
    if mode == TreeUpdateMode.DRY_RUN:
        logger.info(
            f"[DRY RUN] Would apply {fact.fact_type.value if fact.fact_type else 'unknown'}: "
            f"'{fact.original_value}' → '{fact.new_value}' for {person_name}"
        )
        return TreeUpdateResponse(
            result=TreeUpdateResult.SUCCESS,
            operation=TreeOperationType.UPDATE_PERSON,
            person_id=str(fact.people_id),
            message="Dry run - would apply",
        )

    # In INTERACTIVE mode, prompt user
    if mode == TreeUpdateMode.INTERACTIVE:
        should_apply = _prompt_for_approval(fact, person_name)
        if not should_apply:
            return TreeUpdateResponse(
                result=TreeUpdateResult.ALREADY_APPLIED,
                operation=TreeOperationType.UPDATE_PERSON,
                person_id=str(fact.people_id),
                message="Skipped by user",
            )

    # First approve the fact
    success, error = ReviewQueue.approve_suggested_fact(
        db_session=db_session,
        fact_id=fact.id,
        reviewer="action15",
        apply_to_tree=False,  # We'll apply manually for logging
    )

    if not success:
        return TreeUpdateResponse(
            result=TreeUpdateResult.FAILURE,
            operation=TreeOperationType.UPDATE_PERSON,
            person_id=str(fact.people_id),
            message=error or "Failed to approve fact",
        )

    # Apply to tree
    result = tree_service.apply_suggested_fact(db_session, fact, tree_id)

    # Log the operation
    _log_tree_update(
        db_session=db_session,
        fact=fact,
        result=result,
        tree_id=tree_id,
    )

    return result


def _prompt_for_approval(fact: SuggestedFact, person_name: str) -> bool:
    """
    Prompt user for interactive approval.

    Args:
        fact: SuggestedFact to approve
        person_name: Display name of person

    Returns:
        True if user approves, False otherwise
    """
    fact_type = fact.fact_type.value if fact.fact_type else "unknown"
    confidence = f" ({fact.confidence_score}%)" if fact.confidence_score else ""

    print(f"\n{'=' * 60}")
    print(f"📝 Suggested Fact #{fact.id}{confidence}")
    print(f"   Person: {person_name}")
    print(f"   Type: {fact_type}")
    print(f"   Current: {fact.original_value or '(empty)'}")
    print(f"   Proposed: {fact.new_value}")
    print(f"{'=' * 60}")

    while True:
        response = input("Apply this change? [y/n/q] ").strip().lower()
        if response == "y":
            return True
        if response in {"n", "q"}:
            return False
        print("Please enter y (yes), n (no), or q (quit)")


def _log_tree_update(
    db_session: Session,
    fact: SuggestedFact,
    result: TreeUpdateResponse,
    tree_id: str,
) -> None:
    """
    Log tree update operation to database.

    Args:
        db_session: Database session
        fact: The SuggestedFact that was processed
        result: TreeUpdateResponse from the operation
        tree_id: Target tree ID
    """
    status = TreeUpdateStatusEnum.SUCCESS if result.result == TreeUpdateResult.SUCCESS else TreeUpdateStatusEnum.FAILED

    log_entry = TreeUpdateLog(
        suggested_fact_id=fact.id,
        people_id=fact.people_id,
        ancestry_person_id=result.person_id,
        tree_id=tree_id,
        operation_type=result.operation.value,
        api_endpoint="TreeUpdateService.apply_suggested_fact",
        request_payload=None,  # Could serialize fact data
        response_status=200 if status == TreeUpdateStatusEnum.SUCCESS else 400,
        response_body=str(result.api_response)[:1000] if result.api_response else None,
        status=status,
        error_message=result.error_details,
        original_value=fact.original_value,
        new_value=fact.new_value,
    )

    db_session.add(log_entry)
    logger.debug(f"Logged tree update: {log_entry}")


def _log_summary(summary: TreeUpdateSummary) -> None:
    """Log the processing summary."""
    print(f"\n{'=' * 60}")
    print("🌳 Tree Update Summary")
    print(f"{'=' * 60}")
    print(f"   Total Pending: {summary.total_pending}")
    print(f"   Processed: {summary.processed}")
    print(f"   Applied: {summary.applied}")
    print(f"   Skipped: {summary.skipped}")
    print(f"   Failed: {summary.failed}")
    print(f"   Duration: {summary.duration_seconds:.1f}s")

    if summary.by_fact_type:
        print("\n   By Fact Type:")
        for fact_type, count in sorted(summary.by_fact_type.items()):
            print(f"     {fact_type}: {count}")

    if summary.errors:
        print(f"\n   ⚠️ Errors ({len(summary.errors)}):")
        for error in summary.errors[:5]:  # Show first 5
            print(f"     • {error}")
        if len(summary.errors) > 5:
            print(f"     ... and {len(summary.errors) - 5} more")

    print(f"{'=' * 60}")


# =============================================================================
# List and Display Functions
# =============================================================================


def list_pending_facts(
    session_manager: SessionManager,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    List pending SuggestedFacts awaiting review.

    Args:
        session_manager: Active SessionManager
        limit: Maximum number to return

    Returns:
        List of pending facts as dictionaries
    """
    with session_manager.db_transaction() as db_session:
        return ReviewQueue.list_pending_suggested_facts(db_session, limit)


def display_pending_facts(
    session_manager: SessionManager,
    limit: int = 20,
) -> None:
    """
    Display pending SuggestedFacts in a formatted view.

    Args:
        session_manager: Active SessionManager
        limit: Maximum number to display
    """
    facts = list_pending_facts(session_manager, limit)

    if not facts:
        print("📭 No pending suggested facts to review")
        return

    print(f"\n{'=' * 70}")
    print(f"📋 Pending Suggested Facts ({len(facts)} items)")
    print(f"{'=' * 70}")

    for fact in facts:
        confidence = f" ({fact['confidence_score']}%)" if fact.get("confidence_score") else ""
        print(f"\n  [{fact['id']}] {fact['person_name']} - {fact['fact_type']}{confidence}")
        print(f"      Current:  {fact.get('original_value') or '(empty)'}")
        print(f"      Proposed: {fact['new_value']}")

    print(f"\n{'=' * 70}")


# =============================================================================
# Action Entry Point
# =============================================================================


def coord(
    session_manager: SessionManager,
    tree_id: Optional[str] = None,
    mode_str: str = "dry_run",
    limit: int = 20,
) -> bool:
    """
    Main entry point for Action 15.

    Args:
        session_manager: Active SessionManager
        tree_id: Ancestry tree ID (defaults to config)
        mode_str: Mode string (dry_run, apply, interactive)
        limit: Maximum number of facts to process

    Returns:
        True if successful, False otherwise
    """
    # Get tree ID from config if not provided
    if not tree_id:
        tree_id = config_schema.ancestry.tree_id
        if not tree_id:
            logger.error("No tree_id provided and none configured")
            print("❌ Error: No tree ID configured. Set ANCESTRY_TREE_ID in .env")
            return False

    # Parse mode
    try:
        mode = TreeUpdateMode(mode_str)
    except ValueError:
        logger.error(f"Invalid mode: {mode_str}")
        print(f"❌ Invalid mode '{mode_str}'. Use: dry_run, apply, or interactive")
        return False

    # Display pending facts first
    display_pending_facts(session_manager, limit)

    # Confirm if applying
    if mode == TreeUpdateMode.APPLY:
        response = input("\n⚠️ Apply these changes to the tree? [yes/no] ").strip().lower()
        if response != "yes":
            print("Cancelled.")
            return True

    # Run the updates
    summary = run_tree_updates(
        session_manager=session_manager,
        tree_id=tree_id,
        mode=mode,
        limit=limit,
    )

    return summary.failed == 0


# =============================================================================
# Module Tests
# =============================================================================


def _test_tree_update_mode_enum() -> None:
    """Test TreeUpdateMode enum values."""
    assert TreeUpdateMode.DRY_RUN.value == "dry_run"
    assert TreeUpdateMode.APPLY.value == "apply"
    assert TreeUpdateMode.INTERACTIVE.value == "interactive"


def _test_tree_update_summary_creation() -> None:
    """Test TreeUpdateSummary dataclass."""
    summary = TreeUpdateSummary(
        total_pending=10,
        processed=8,
        applied=6,
        skipped=1,
        failed=1,
    )
    assert summary.total_pending == 10
    assert summary.applied == 6
    assert summary.by_fact_type == {}


def _test_tree_update_summary_with_fact_types() -> None:
    """Test TreeUpdateSummary with fact type tracking."""
    summary = TreeUpdateSummary()
    summary.by_fact_type["birth"] = 3
    summary.by_fact_type["death"] = 2
    assert sum(summary.by_fact_type.values()) == 5


def module_tests() -> bool:
    """Run module tests."""
    suite = TestSuite("Action 15: Tree Updates", "actions/action15_tree_updates.py")
    suite.start_suite()

    suite.run_test(
        "TreeUpdateMode enum values",
        _test_tree_update_mode_enum,
        "Verifies TreeUpdateMode enum has expected values",
    )
    suite.run_test(
        "TreeUpdateSummary creation",
        _test_tree_update_summary_creation,
        "Verifies TreeUpdateSummary dataclass instantiation",
    )
    suite.run_test(
        "TreeUpdateSummary fact type tracking",
        _test_tree_update_summary_with_fact_types,
        "Verifies fact type tracking in summary",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys

    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)

    # Otherwise run tests by default when executed directly
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
