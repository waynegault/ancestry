#!/usr/bin/env python3
"""
Shadow Mode Analyzer for MessageSendOrchestrator

Phase 4.3: Compares orchestrator decisions against legacy action decisions
without affecting actual message sends. Logs discrepancies for analysis.

Usage:
    1. Enable shadow mode: SHADOW_MODE_ENABLED=true in .env
    2. Run normal operations (Action 8, 9, 11)
    3. Analyze discrepancies: python -m messaging.shadow_mode_analyzer --report
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

if TYPE_CHECKING:
    from core.session_manager import SessionManager
    from messaging.send_orchestrator import MessageSendContext, SendDecision

logger = logging.getLogger(__name__)


# =============================================================================
# Shadow Mode Configuration
# =============================================================================

SHADOW_LOG_PATH = Path("Logs/shadow_mode_decisions.jsonl")
DISCREPANCY_LOG_PATH = Path("Logs/shadow_mode_discrepancies.jsonl")


@dataclass
class LegacyDecision:
    """Represents a decision made by legacy action code."""

    action_name: str
    should_send: bool
    block_reason: str | None = None
    person_id: int | None = None
    trigger_type: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class ShadowComparison:
    """Result of comparing legacy vs orchestrator decisions."""

    person_id: int
    action_name: str
    legacy_should_send: bool
    legacy_block_reason: str | None
    orchestrator_should_send: bool
    orchestrator_block_reason: str | None
    is_match: bool
    discrepancy_type: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


# =============================================================================
# Shadow Mode Analyzer
# =============================================================================


class ShadowModeAnalyzer:
    """
    Analyzes orchestrator decisions in shadow mode without affecting sends.

    In shadow mode:
    1. Legacy code makes actual send decisions
    2. Orchestrator runs in parallel (no actual sends)
    3. Decisions are compared and discrepancies logged
    """

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        """Initialize analyzer."""
        self._session_manager = session_manager
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._shadow_enabled = self._check_shadow_mode_enabled()

    @staticmethod
    def _check_shadow_mode_enabled() -> bool:
        """Check if shadow mode is enabled via feature flag."""
        try:
            from config.config_manager import get_config_manager

            config_schema = get_config_manager().get_config()
            return getattr(config_schema, "shadow_mode_enabled", False)
        except (ImportError, AttributeError):
            return False

    @property
    def is_enabled(self) -> bool:
        """Check if shadow mode is currently enabled."""
        return self._shadow_enabled

    def compare_decisions(
        self,
        legacy_decision: LegacyDecision,
        orchestrator_decision: SendDecision,
        context: MessageSendContext,
    ) -> ShadowComparison:
        """
        Compare a legacy decision with an orchestrator decision.

        Args:
            legacy_decision: Decision from legacy action code
            orchestrator_decision: Decision from MessageSendOrchestrator
            context: The message send context

        Returns:
            ShadowComparison with match status and discrepancy details
        """
        is_match = legacy_decision.should_send == orchestrator_decision.should_send

        discrepancy_type = None
        if not is_match:
            if legacy_decision.should_send and not orchestrator_decision.should_send:
                discrepancy_type = "orchestrator_more_restrictive"
            else:
                discrepancy_type = "orchestrator_more_permissive"

        comparison = ShadowComparison(
            person_id=context.person.id,
            action_name=legacy_decision.action_name,
            legacy_should_send=legacy_decision.should_send,
            legacy_block_reason=legacy_decision.block_reason,
            orchestrator_should_send=orchestrator_decision.should_send,
            orchestrator_block_reason=orchestrator_decision.block_reason,
            is_match=is_match,
            discrepancy_type=discrepancy_type,
        )

        # Log the comparison
        self._log_comparison(comparison)

        if not is_match:
            self._log_discrepancy(comparison)
            self._logger.warning(
                f"🔍 Shadow mode discrepancy: {legacy_decision.action_name} "
                f"person_id={context.person.id} - "
                f"legacy={legacy_decision.should_send}, "
                f"orchestrator={orchestrator_decision.should_send}"
            )

        return comparison

    def _log_comparison(self, comparison: ShadowComparison) -> None:
        """Log comparison to shadow log file."""
        try:
            SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with SHADOW_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(comparison.to_dict()) + "\n")
        except OSError as e:
            self._logger.error(f"Failed to write shadow log: {e}")

    def _log_discrepancy(self, comparison: ShadowComparison) -> None:
        """Log discrepancy to separate discrepancy log."""
        try:
            DISCREPANCY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DISCREPANCY_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(comparison.to_dict()) + "\n")
        except OSError as e:
            self._logger.error(f"Failed to write discrepancy log: {e}")

    def run_shadow_check(
        self,
        context: MessageSendContext,
        legacy_decision: LegacyDecision,
    ) -> ShadowComparison:
        """
        Run orchestrator in shadow mode and compare with legacy decision.

        Args:
            context: The message send context
            legacy_decision: Decision already made by legacy code

        Returns:
            ShadowComparison result
        """
        from messaging.send_orchestrator import MessageSendOrchestrator

        if self._session_manager is None:
            raise ValueError("SessionManager required for shadow check")

        # Create orchestrator and get its decision (without sending)
        orchestrator = MessageSendOrchestrator(self._session_manager)
        orchestrator_decision = orchestrator._make_decision(context)

        return self.compare_decisions(legacy_decision, orchestrator_decision, context)


# =============================================================================
# Report Generation
# =============================================================================


def generate_discrepancy_report() -> dict[str, Any]:
    """
    Generate a summary report of all discrepancies.

    Returns:
        Dictionary with report statistics and details
    """
    if not DISCREPANCY_LOG_PATH.exists():
        return {
            "status": "no_data",
            "message": "No discrepancy log found. Run shadow mode first.",
            "total_discrepancies": 0,
        }

    discrepancies: list[dict[str, Any]] = []
    with DISCREPANCY_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                discrepancies.append(json.loads(line))

    if not discrepancies:
        return {
            "status": "clean",
            "message": "No discrepancies found!",
            "total_discrepancies": 0,
        }

    # Analyze discrepancies
    by_action: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_block_reason: dict[str, int] = {}

    for d in discrepancies:
        action = d.get("action_name", "unknown")
        by_action[action] = by_action.get(action, 0) + 1

        disc_type = d.get("discrepancy_type", "unknown")
        by_type[disc_type] = by_type.get(disc_type, 0) + 1

        # Track orchestrator block reasons when it's more restrictive
        if disc_type == "orchestrator_more_restrictive":
            reason = d.get("orchestrator_block_reason", "unknown")
            by_block_reason[reason] = by_block_reason.get(reason, 0) + 1

    return {
        "status": "discrepancies_found",
        "total_discrepancies": len(discrepancies),
        "by_action": by_action,
        "by_type": by_type,
        "orchestrator_block_reasons": by_block_reason,
        "sample_discrepancies": discrepancies[:5],  # First 5 for review
    }


def generate_shadow_mode_stats() -> dict[str, Any]:
    """
    Generate statistics from shadow mode log.

    Returns:
        Dictionary with overall shadow mode statistics
    """
    if not SHADOW_LOG_PATH.exists():
        return {
            "status": "no_data",
            "message": "No shadow log found. Enable shadow mode first.",
        }

    total = 0
    matches = 0
    mismatches = 0

    with SHADOW_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                total += 1
                if entry.get("is_match"):
                    matches += 1
                else:
                    mismatches += 1

    match_rate = (matches / total * 100) if total > 0 else 0

    return {
        "status": "ok",
        "total_comparisons": total,
        "matches": matches,
        "mismatches": mismatches,
        "match_rate_percent": round(match_rate, 2),
        "ready_for_cutover": match_rate >= 99.0 and total >= 100,
    }


def print_report() -> None:
    """Print formatted discrepancy report to console."""
    stats = generate_shadow_mode_stats()
    report = generate_discrepancy_report()

    print("\n" + "=" * 60)
    print("📊 SHADOW MODE ANALYSIS REPORT")
    print("=" * 60)

    print("\n📈 Overall Statistics:")
    print(f"   Total comparisons: {stats.get('total_comparisons', 0)}")
    print(f"   Matches: {stats.get('matches', 0)}")
    print(f"   Mismatches: {stats.get('mismatches', 0)}")
    print(f"   Match rate: {stats.get('match_rate_percent', 0)}%")
    print(f"   Ready for cutover: {'✅ Yes' if stats.get('ready_for_cutover') else '❌ No'}")

    if report.get("total_discrepancies", 0) > 0:
        print("\n⚠️  Discrepancy Breakdown:")
        print(f"   Total discrepancies: {report['total_discrepancies']}")

        print("\n   By Action:")
        for action, count in report.get("by_action", {}).items():
            print(f"      {action}: {count}")

        print("\n   By Type:")
        for dtype, count in report.get("by_type", {}).items():
            print(f"      {dtype}: {count}")

        if report.get("orchestrator_block_reasons"):
            print("\n   Orchestrator Block Reasons:")
            for reason, count in report.get("orchestrator_block_reasons", {}).items():
                print(f"      {reason}: {count}")

    print("\n" + "=" * 60)


# =============================================================================
# Module Tests
# =============================================================================


def _test_legacy_decision_creation() -> None:
    """Test LegacyDecision dataclass creation."""
    decision = LegacyDecision(
        action_name="Action8",
        should_send=True,
        person_id=123,
    )
    assert decision.action_name == "Action8"
    assert decision.should_send is True
    assert decision.person_id == 123


def _test_legacy_decision_to_dict() -> None:
    """Test LegacyDecision serialization."""
    decision = LegacyDecision(
        action_name="Action9",
        should_send=False,
        block_reason="Person opted out",
    )
    result = decision.to_dict()
    assert result["action_name"] == "Action9"
    assert result["should_send"] is False
    assert "timestamp" in result


def _test_shadow_comparison_match() -> None:
    """Test ShadowComparison for matching decisions."""
    comparison = ShadowComparison(
        person_id=1,
        action_name="Action8",
        legacy_should_send=True,
        legacy_block_reason=None,
        orchestrator_should_send=True,
        orchestrator_block_reason=None,
        is_match=True,
    )
    assert comparison.is_match is True
    assert comparison.discrepancy_type is None


def _test_shadow_comparison_discrepancy() -> None:
    """Test ShadowComparison for mismatched decisions."""
    comparison = ShadowComparison(
        person_id=1,
        action_name="Action8",
        legacy_should_send=True,
        legacy_block_reason=None,
        orchestrator_should_send=False,
        orchestrator_block_reason="Duplicate prevention",
        is_match=False,
        discrepancy_type="orchestrator_more_restrictive",
    )
    assert comparison.is_match is False
    assert comparison.discrepancy_type == "orchestrator_more_restrictive"


def _test_analyzer_initialization() -> None:
    """Test ShadowModeAnalyzer initialization."""
    analyzer = ShadowModeAnalyzer()
    # Should initialize without session_manager for basic operations
    assert analyzer is not None
    # Shadow mode disabled by default (no config)
    assert analyzer.is_enabled is False


def _test_report_generation_no_data() -> None:
    """Test report generation with no data."""
    # Temporarily use non-existent path
    original_path = DISCREPANCY_LOG_PATH
    globals()["DISCREPANCY_LOG_PATH"] = Path("Logs/nonexistent_test_file.jsonl")

    try:
        report = generate_discrepancy_report()
        assert report["status"] == "no_data"
        assert report["total_discrepancies"] == 0
    finally:
        globals()["DISCREPANCY_LOG_PATH"] = original_path


def _test_stats_generation_no_data() -> None:
    """Test stats generation with no data."""
    original_path = SHADOW_LOG_PATH
    globals()["SHADOW_LOG_PATH"] = Path("Logs/nonexistent_test_file.jsonl")

    try:
        stats = generate_shadow_mode_stats()
        assert stats["status"] == "no_data"
    finally:
        globals()["SHADOW_LOG_PATH"] = original_path


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Shadow Mode Analyzer", "messaging/shadow_mode_analyzer.py")
    suite.start_suite()

    suite.run_test(
        test_name="LegacyDecision creation",
        test_func=_test_legacy_decision_creation,
        test_summary="Create LegacyDecision dataclass",
        expected_outcome="All fields set correctly",
    )

    suite.run_test(
        test_name="LegacyDecision to_dict",
        test_func=_test_legacy_decision_to_dict,
        test_summary="Serialize LegacyDecision to dictionary",
        expected_outcome="Dictionary contains all fields with ISO timestamp",
    )

    suite.run_test(
        test_name="ShadowComparison match",
        test_func=_test_shadow_comparison_match,
        test_summary="Create matching ShadowComparison",
        expected_outcome="is_match=True, no discrepancy_type",
    )

    suite.run_test(
        test_name="ShadowComparison discrepancy",
        test_func=_test_shadow_comparison_discrepancy,
        test_summary="Create mismatched ShadowComparison",
        expected_outcome="is_match=False, discrepancy_type set",
    )

    suite.run_test(
        test_name="Analyzer initialization",
        test_func=_test_analyzer_initialization,
        test_summary="Initialize ShadowModeAnalyzer",
        expected_outcome="Analyzer created, shadow mode disabled by default",
    )

    suite.run_test(
        test_name="Report generation - no data",
        test_func=_test_report_generation_no_data,
        test_summary="Generate report with no discrepancy log",
        expected_outcome="Returns no_data status",
    )

    suite.run_test(
        test_name="Stats generation - no data",
        test_func=_test_stats_generation_no_data,
        test_summary="Generate stats with no shadow log",
        expected_outcome="Returns no_data status",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import os
    if os.environ.get("RUN_MODULE_TESTS") == "1":
        sys.exit(0 if run_comprehensive_tests() else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        print_report()
    else:
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
