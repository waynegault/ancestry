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


import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from actions.action_review import ReviewQueue
from api.tree_update import (
    TreeOperationType,
    TreeUpdateResponse,
    TreeUpdateResult,
    TreeUpdateService,
)
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
    fact_types: list[FactTypeEnum] | None = None,
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

    with session_manager.db_manager.get_session_context() as db_session:
        assert db_session is not None, "Failed to get database session"
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
    with session_manager.db_manager.get_session_context() as db_session:
        assert db_session is not None, "Failed to get database session"
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
    tree_id: str | None = None,
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
        tree_id = session_manager.my_tree_id
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
    """Test TreeUpdateMode enum values and construction from strings."""
    # Verify all enum values
    assert TreeUpdateMode.DRY_RUN.value == "dry_run"
    assert TreeUpdateMode.APPLY.value == "apply"
    assert TreeUpdateMode.INTERACTIVE.value == "interactive"

    # Verify enum construction from string values (used by coord())
    assert TreeUpdateMode("dry_run") is TreeUpdateMode.DRY_RUN
    assert TreeUpdateMode("apply") is TreeUpdateMode.APPLY
    assert TreeUpdateMode("interactive") is TreeUpdateMode.INTERACTIVE

    # Verify invalid mode string raises ValueError (coord() relies on this)
    try:
        TreeUpdateMode("invalid_mode")
        raise AssertionError("Expected ValueError for invalid mode string")
    except ValueError:
        pass

    # Verify all members are accounted for
    assert len(TreeUpdateMode) == 3, f"Expected 3 modes, got {len(TreeUpdateMode)}"


def _test_tree_update_summary_tracking() -> None:
    """Test TreeUpdateSummary tracks counts and computes correctly."""
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

    # Verify derived relationships: processed == applied + skipped + failed
    assert summary.processed == summary.applied + summary.skipped + summary.failed

    # Verify unprocessed count
    unprocessed = summary.total_pending - summary.processed
    assert unprocessed == 2, f"Expected 2 unprocessed, got {unprocessed}"

    # Verify errors list is mutable and independent per instance
    summary.errors.append("test error")
    assert len(summary.errors) == 1
    other = TreeUpdateSummary()
    assert len(other.errors) == 0, "Errors list should not be shared between instances"


def _test_tree_update_summary_fact_type_aggregation() -> None:
    """Test TreeUpdateSummary fact type tracking mirrors real processing logic."""
    summary = TreeUpdateSummary()

    # Simulate the processing loop from run_tree_updates
    fact_types_processed = ["birth", "birth", "death", "birth", "marriage", "death"]
    for ft in fact_types_processed:
        summary.by_fact_type[ft] = summary.by_fact_type.get(ft, 0) + 1
        summary.processed += 1

    assert summary.by_fact_type["birth"] == 3
    assert summary.by_fact_type["death"] == 2
    assert summary.by_fact_type["marriage"] == 1
    assert sum(summary.by_fact_type.values()) == 6
    assert summary.processed == 6

    # Verify sorted output order (used by _log_summary)
    sorted_types = sorted(summary.by_fact_type.items())
    assert sorted_types[0][0] == "birth"
    assert sorted_types[1][0] == "death"
    assert sorted_types[2][0] == "marriage"


def _test_dry_run_returns_success_without_mutations() -> None:
    """Test that _process_single_fact in DRY_RUN mode returns success without calling tree service."""
    from unittest.mock import MagicMock

    mock_db_session = MagicMock(spec=Session)
    mock_tree_service = MagicMock(spec=TreeUpdateService)
    mock_fact = MagicMock(spec=SuggestedFact)
    mock_fact.id = 42
    mock_fact.people_id = 100
    mock_fact.fact_type = FactTypeEnum.BIRTH
    mock_fact.original_value = "1 Jan 1900"
    mock_fact.new_value = "1 January 1900"

    # Provide a person lookup result
    mock_person = MagicMock(spec=Person)
    mock_person.display_name = "John Smith"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_person

    result = _process_single_fact(
        db_session=mock_db_session,
        tree_service=mock_tree_service,
        fact=mock_fact,
        tree_id="tree-123",
        mode=TreeUpdateMode.DRY_RUN,
    )

    # DRY_RUN should return SUCCESS without calling tree_service.apply_suggested_fact
    assert result.result == TreeUpdateResult.SUCCESS, f"Expected SUCCESS, got {result.result}"
    assert "Dry run" in result.message
    mock_tree_service.apply_suggested_fact.assert_not_called()


def _test_run_tree_updates_feature_flag_disabled() -> None:
    """Test run_tree_updates returns early with error when feature flag is disabled."""
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()

    with patch(f"{__name__}.is_feature_enabled", return_value=False):
        summary = run_tree_updates(
            session_manager=mock_sm,
            tree_id="tree-123",
            mode=TreeUpdateMode.DRY_RUN,
        )

    assert summary.applied == 0, "Should not apply anything when feature flag is off"
    assert summary.failed == 0
    assert summary.total_pending == 0
    assert "Feature flag disabled" in summary.errors
    assert summary.duration_seconds >= 0
    # DB should never be accessed
    mock_sm.db_manager.get_session_context.assert_not_called()


def _test_run_tree_updates_empty_pending() -> None:
    """Test run_tree_updates with no pending facts returns clean summary."""
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value.limit.return_value.all.return_value = []
    mock_db.query.return_value = mock_query

    @contextmanager
    def fake_session_ctx():
        yield mock_db

    mock_sm.db_manager.get_session_context = fake_session_ctx

    with patch(f"{__name__}.is_feature_enabled", return_value=True):
        summary = run_tree_updates(
            session_manager=mock_sm,
            tree_id="tree-123",
            mode=TreeUpdateMode.DRY_RUN,
        )

    assert summary.total_pending == 0
    assert summary.processed == 0
    assert summary.applied == 0
    assert summary.duration_seconds >= 0


def _test_run_tree_updates_dry_run_processes_facts() -> None:
    """Test run_tree_updates in DRY_RUN mode processes facts as SUCCESS but never mutates tree."""
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    mock_sm = MagicMock()
    mock_db = MagicMock()

    # Create two mock SuggestedFact objects
    mock_fact1 = MagicMock(spec=SuggestedFact)
    mock_fact1.id = 1
    mock_fact1.people_id = 10
    mock_fact1.fact_type = FactTypeEnum.BIRTH
    mock_fact1.original_value = "1900"
    mock_fact1.new_value = "1 Jan 1900"

    mock_fact2 = MagicMock(spec=SuggestedFact)
    mock_fact2.id = 2
    mock_fact2.people_id = 20
    mock_fact2.fact_type = FactTypeEnum.DEATH
    mock_fact2.original_value = "1970"
    mock_fact2.new_value = "15 Mar 1970"

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value.limit.return_value.all.return_value = [mock_fact1, mock_fact2]
    mock_db.query.return_value = mock_query
    # Person lookup
    mock_person = MagicMock()
    mock_person.display_name = "Test Person"
    mock_db.query.return_value.filter.return_value.first.return_value = mock_person

    @contextmanager
    def fake_session_ctx():
        yield mock_db

    mock_sm.db_manager.get_session_context = fake_session_ctx

    mock_tree_service = MagicMock(spec=TreeUpdateService)

    with (
        patch(f"{__name__}.is_feature_enabled", return_value=True),
        patch(f"{__name__}.TreeUpdateService", return_value=mock_tree_service),
    ):
        summary = run_tree_updates(
            session_manager=mock_sm,
            tree_id="tree-123",
            mode=TreeUpdateMode.DRY_RUN,
        )

    assert summary.total_pending == 2
    assert summary.processed == 2
    assert summary.applied == 2  # DRY_RUN still counts as SUCCESS/applied
    assert summary.failed == 0
    assert "BIRTH" in summary.by_fact_type
    assert "DEATH" in summary.by_fact_type
    # Tree service apply should never be called in DRY_RUN
    mock_tree_service.apply_suggested_fact.assert_not_called()


def _test_log_summary_output() -> None:
    """Test _log_summary produces correctly formatted output."""
    import io
    import sys

    summary = TreeUpdateSummary(
        total_pending=5,
        processed=5,
        applied=3,
        skipped=1,
        failed=1,
        by_fact_type={"birth": 2, "death": 1, "marriage": 2},
        errors=["Fact 10: API timeout"],
        duration_seconds=12.345,
    )

    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        _log_summary(summary)
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()

    # Verify key content in output
    assert "Tree Update Summary" in output
    assert "Total Pending: 5" in output
    assert "Applied: 3" in output
    assert "Skipped: 1" in output
    assert "Failed: 1" in output
    assert "12.3" in output  # Duration formatted to 1 decimal
    assert "birth: 2" in output
    assert "death: 1" in output
    assert "marriage: 2" in output
    assert "Fact 10: API timeout" in output


def _test_log_summary_truncates_long_error_list() -> None:
    """Test _log_summary only shows first 5 errors and indicates remaining count."""
    import io
    import sys

    summary = TreeUpdateSummary(
        total_pending=10,
        processed=10,
        failed=8,
        errors=[f"Error {i}" for i in range(8)],
    )

    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        _log_summary(summary)
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "Error 0" in output
    assert "Error 4" in output
    # Error 5-7 should be truncated
    assert "Error 5" not in output
    assert "and 3 more" in output


def _test_coord_rejects_invalid_mode() -> None:
    """Test coord() returns False for invalid mode strings."""
    from unittest.mock import MagicMock

    mock_sm = MagicMock(spec=SessionManager)
    mock_sm.my_tree_id = "tree-123"

    result = coord(mock_sm, tree_id="tree-123", mode_str="invalid_mode")
    assert result is False, "coord() should return False for invalid mode"


def _test_summary_dataclass_defaults() -> None:
    """Test TreeUpdateSummary default values and mutable default isolation."""
    s1 = TreeUpdateSummary()
    s2 = TreeUpdateSummary()

    # Defaults should be zero/empty
    assert s1.total_pending == 0
    assert s1.processed == 0
    assert s1.applied == 0
    assert s1.skipped == 0
    assert s1.failed == 0
    assert s1.duration_seconds == 0.0
    assert s1.by_fact_type == {}
    assert s1.errors == []

    # Mutating one instance must not affect the other (mutable default isolation)
    s1.by_fact_type["birth"] = 5
    s1.errors.append("oops")
    assert s2.by_fact_type == {}, "Mutable defaults must be independent"
    assert s2.errors == [], "Mutable defaults must be independent"


def module_tests() -> bool:
    """Run module tests."""
    suite = TestSuite("Action 15: Tree Updates", "actions/action15_tree_updates.py")
    suite.start_suite()

    suite.run_test(
        "TreeUpdateMode enum values and string construction",
        _test_tree_update_mode_enum,
        "Verifies enum values, construction from strings, and invalid value rejection",
    )
    suite.run_test(
        "TreeUpdateSummary count tracking",
        _test_tree_update_summary_tracking,
        "Verifies summary tracks applied/skipped/failed and derived counts",
    )
    suite.run_test(
        "TreeUpdateSummary fact type aggregation",
        _test_tree_update_summary_fact_type_aggregation,
        "Verifies fact type counting mirrors real processing logic",
    )
    suite.run_test(
        "TreeUpdateSummary defaults and mutable isolation",
        _test_summary_dataclass_defaults,
        "Verifies default values and mutable field independence between instances",
    )
    suite.run_test(
        "DRY_RUN _process_single_fact returns SUCCESS without mutations",
        _test_dry_run_returns_success_without_mutations,
        "Invokes _process_single_fact in DRY_RUN and verifies tree service is NOT called",
    )
    suite.run_test(
        "run_tree_updates returns early when feature flag disabled",
        _test_run_tree_updates_feature_flag_disabled,
        "Invokes run_tree_updates with disabled flag and verifies early exit with error",
    )
    suite.run_test(
        "run_tree_updates with empty pending list",
        _test_run_tree_updates_empty_pending,
        "Verifies clean summary when no pending facts exist",
    )
    suite.run_test(
        "run_tree_updates DRY_RUN processes facts without tree mutations",
        _test_run_tree_updates_dry_run_processes_facts,
        "Invokes run_tree_updates with mock facts in DRY_RUN and verifies counts",
    )
    suite.run_test(
        "_log_summary produces correct formatted output",
        _test_log_summary_output,
        "Captures stdout and verifies summary formatting including fact types and errors",
    )
    suite.run_test(
        "_log_summary truncates long error lists",
        _test_log_summary_truncates_long_error_list,
        "Verifies only first 5 errors shown with remainder count",
    )
    suite.run_test(
        "coord() rejects invalid mode strings",
        _test_coord_rejects_invalid_mode,
        "Verifies coord returns False for unrecognized mode value",
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
