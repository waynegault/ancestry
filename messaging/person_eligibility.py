"""
Person Eligibility Checker for Unified Message Send Orchestrator.

This module centralizes all person-level eligibility checks that determine
whether a person can receive a message. Extracted from actions/action8_messaging.py
as part of the unified messaging refactoring (Phase 2.1).

Responsibilities:
    - Status-based eligibility (ARCHIVE, BLOCKED, DEAD)
    - Opt-out detection integration
    - In-tree vs out-tree classification
    - Rate limiting per-person

Usage:
    from messaging.person_eligibility import PersonEligibilityChecker

    checker = PersonEligibilityChecker(db_session)
    result = checker.check_eligibility(person, context)
    if not result.is_eligible:
        logger.info(f"Person not eligible: {result.reason}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, cast

from testing.test_framework import TestSuite

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from core.database import ConversationLog, Person

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================


class IneligibilityReason(Enum):
    """Reasons why a person may be ineligible for messaging."""

    NONE = auto()  # Person is eligible
    STATUS_ARCHIVED = auto()  # Person status is ARCHIVE
    STATUS_BLOCKED = auto()  # Person status is BLOCKED
    STATUS_DEAD = auto()  # Person status is DEAD
    STATUS_DESIST = auto()  # Person opted out (DESIST status)
    OPT_OUT_DETECTED = auto()  # Opt-out signal in latest message
    RATE_LIMITED = auto()  # Too many messages recently
    NO_CONTACT_INFO = auto()  # No way to contact (no profile_id)
    TREE_STATUS_CHANGED = auto()  # Recently added to tree, special handling


class TreeClassification(Enum):
    """Classification of person's relationship to the family tree."""

    IN_TREE = auto()  # Person is in the family tree
    OUT_OF_TREE = auto()  # Person is not in the family tree
    RECENTLY_ADDED = auto()  # Person was recently added to tree


# ============================================================
# Data Classes
# ============================================================


@dataclass
class EligibilityResult:
    """Result of person eligibility check."""

    is_eligible: bool
    reason: IneligibilityReason = IneligibilityReason.NONE
    reason_detail: str = ""
    tree_classification: TreeClassification = TreeClassification.OUT_OF_TREE
    can_send_special_message: bool = False  # For recently-added-to-tree cases


@dataclass
class RateLimitConfig:
    """Configuration for per-person rate limiting."""

    max_messages_per_window: int = 3
    window_hours: int = 72  # 3 days
    cooldown_hours: int = 24  # Minimum time between messages


@dataclass
class PersonEligibilityContext:
    """Additional context for eligibility checks."""

    conversation_logs: list[ConversationLog] = field(default_factory=list)
    latest_inbound_message: Optional[str] = None
    check_opt_out: bool = True
    check_rate_limit: bool = True
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)


# ============================================================
# PersonEligibilityChecker Class
# ============================================================


class PersonEligibilityChecker:
    """
    Centralized eligibility checker for messaging a person.

    Consolidates eligibility logic previously scattered across:
    - action8_messaging.py: _check_person_eligibility(), _should_skip_for_opt_out()
    - action11_send_approved_drafts.py: Hard stop checks
    - send_orchestrator.py: Safety checks

    Args:
        db_session: Optional SQLAlchemy session for database queries

    Example:
        checker = PersonEligibilityChecker(db_session)
        result = checker.check_eligibility(person, context)
        if not result.is_eligible:
            print(f"Cannot message {person.username}: {result.reason.name}")
    """

    # Statuses that prevent messaging
    BLOCKED_STATUSES = frozenset(["ARCHIVE", "BLOCKED", "DEAD"])
    OPT_OUT_STATUSES = frozenset(["DESIST"])

    def __init__(self, db_session: Optional[Session] = None) -> None:
        """Initialize with optional database session."""
        self._db_session = db_session
        self._opt_out_detector: Optional[object] = None

    def check_eligibility(
        self,
        person: Person,
        context: Optional[PersonEligibilityContext] = None,
    ) -> EligibilityResult:
        """
        Check if a person is eligible to receive a message.

        Performs checks in order of priority:
        1. Status-based (ARCHIVE, BLOCKED, DEAD, DESIST)
        2. Contact info availability
        3. Opt-out detection (if enabled)
        4. Rate limiting (if enabled)
        5. Tree classification

        Args:
            person: Person to check eligibility for
            context: Optional additional context for checks

        Returns:
            EligibilityResult with is_eligible status and reason
        """
        ctx = context or PersonEligibilityContext()

        # 1. Check status-based eligibility
        status_result = self._check_status_eligibility(person)
        if not status_result.is_eligible:
            return status_result

        # 2. Check contact info
        contact_result = self._check_contact_info(person)
        if not contact_result.is_eligible:
            return contact_result

        # 3. Check opt-out (if enabled)
        if ctx.check_opt_out:
            opt_out_result = self._check_opt_out(person, ctx)
            if not opt_out_result.is_eligible:
                return opt_out_result

        # 4. Check rate limits (if enabled)
        if ctx.check_rate_limit:
            rate_result = self._check_rate_limit(person, ctx)
            if not rate_result.is_eligible:
                return rate_result

        # 5. Determine tree classification
        tree_classification = self._classify_tree_status(person)

        # All checks passed - person is eligible
        return EligibilityResult(
            is_eligible=True,
            reason=IneligibilityReason.NONE,
            reason_detail="",
            tree_classification=tree_classification,
            can_send_special_message=(tree_classification == TreeClassification.RECENTLY_ADDED),
        )

    def _check_status_eligibility(self, person: Person) -> EligibilityResult:
        """Check if person status allows messaging."""
        person_status = self._get_status_name(person)

        if person_status in self.BLOCKED_STATUSES:
            reason_map = {
                "ARCHIVE": IneligibilityReason.STATUS_ARCHIVED,
                "BLOCKED": IneligibilityReason.STATUS_BLOCKED,
                "DEAD": IneligibilityReason.STATUS_DEAD,
            }
            return EligibilityResult(
                is_eligible=False,
                reason=reason_map.get(person_status, IneligibilityReason.STATUS_ARCHIVED),
                reason_detail=f"Person status is {person_status}",
            )

        if person_status in self.OPT_OUT_STATUSES:
            return EligibilityResult(
                is_eligible=False,
                reason=IneligibilityReason.STATUS_DESIST,
                reason_detail="Person has opted out (DESIST status)",
            )

        return EligibilityResult(is_eligible=True)

    def _check_contact_info(self, person: Person) -> EligibilityResult:  # noqa: PLR6301
        """Check if person has contact information."""
        # Need either profile_id or administrator_profile_id to message
        profile_id = getattr(person, "profile_id", None)
        admin_profile_id = getattr(person, "administrator_profile_id", None)

        if not profile_id and not admin_profile_id:
            return EligibilityResult(
                is_eligible=False,
                reason=IneligibilityReason.NO_CONTACT_INFO,
                reason_detail="Person has no profile_id or administrator_profile_id",
            )

        return EligibilityResult(is_eligible=True)

    def _check_opt_out(
        self,
        person: Person,
        context: PersonEligibilityContext,
    ) -> EligibilityResult:
        """Check for opt-out signals in conversation."""
        if not self._db_session:
            # No database session - skip opt-out check
            return EligibilityResult(is_eligible=True)

        try:
            # Lazy import to avoid circular dependencies
            from messaging.opt_out_detection import OptOutDetector

            detector = OptOutDetector(self._db_session)

            # Check person's opt-out status
            person_id = getattr(person, "id", None)
            if person_id:
                can_send: bool
                reason: str
                validate_fn = getattr(detector, "validate_can_send", None)
                if validate_fn:
                    can_send, reason = validate_fn(person_id)
                else:
                    can_send, reason = True, ""
                if not can_send:
                    return EligibilityResult(
                        is_eligible=False,
                        reason=IneligibilityReason.OPT_OUT_DETECTED,
                        reason_detail=f"Opt-out status: {reason}",
                    )

            # Check latest inbound message for opt-out signals
            if context.latest_inbound_message:
                analyze_fn = getattr(detector, "analyze_message", None)
                analysis: Any = analyze_fn(context.latest_inbound_message) if analyze_fn else None
                is_opt_out: bool = getattr(analysis, "is_opt_out", False)
                if is_opt_out:
                    confidence: float = getattr(analysis, "confidence", 0.0)
                    suggested_action: str = getattr(analysis, "suggested_action", "unknown")
                    return EligibilityResult(
                        is_eligible=False,
                        reason=IneligibilityReason.OPT_OUT_DETECTED,
                        reason_detail=(
                            f"Opt-out detected in message (confidence={confidence:.2f}, action={suggested_action})"
                        ),
                    )
        except Exception as e:
            logger.warning(f"Opt-out check failed: {e}")
            # Fail open - allow messaging if check fails

        return EligibilityResult(is_eligible=True)

    def _check_rate_limit(
        self,
        _person: Person,  # Reserved for future per-person rate tracking
        context: PersonEligibilityContext,
    ) -> EligibilityResult:
        """Check if person has been messaged too recently or too often."""
        config = context.rate_limit_config
        logs = context.conversation_logs

        if not logs:
            return EligibilityResult(is_eligible=True)

        now = datetime.now()
        outbound_count, most_recent = self._count_outbound_messages(logs, now - timedelta(hours=config.window_hours))

        # Check cooldown
        cooldown_result = self._check_cooldown(now, most_recent, config)
        if cooldown_result:
            return cooldown_result

        # Check max messages in window
        if outbound_count >= config.max_messages_per_window:
            return EligibilityResult(
                is_eligible=False,
                reason=IneligibilityReason.RATE_LIMITED,
                reason_detail=(
                    f"Already sent {outbound_count} messages in last "
                    f"{config.window_hours}h (max: {config.max_messages_per_window})"
                ),
            )

        return EligibilityResult(is_eligible=True)

    @staticmethod
    def _count_outbound_messages(
        logs: list[ConversationLog],
        window_start: datetime,
    ) -> tuple[int, Optional[datetime]]:
        """Count outbound messages and find most recent."""
        outbound_count = 0
        most_recent: Optional[datetime] = None

        for log in logs:
            log_direction = getattr(log, "direction", None)
            log_timestamp = getattr(log, "timestamp", None)

            if log_direction == "outbound" and log_timestamp:
                if log_timestamp > window_start:
                    outbound_count += 1
                if most_recent is None or log_timestamp > most_recent:
                    most_recent = log_timestamp

        return outbound_count, most_recent

    @staticmethod
    def _check_cooldown(
        now: datetime,
        most_recent: Optional[datetime],
        config: RateLimitConfig,
    ) -> Optional[EligibilityResult]:
        """Check if cooldown period has passed."""
        if not most_recent:
            return None

        cooldown_threshold = now - timedelta(hours=config.cooldown_hours)
        if most_recent > cooldown_threshold:
            hours_since = (now - most_recent).total_seconds() / 3600
            return EligibilityResult(
                is_eligible=False,
                reason=IneligibilityReason.RATE_LIMITED,
                reason_detail=f"Last message sent {hours_since:.1f}h ago (cooldown: {config.cooldown_hours}h)",
            )
        return None

    def _classify_tree_status(self, person: Person) -> TreeClassification:
        """Classify person's relationship to the family tree."""
        in_my_tree = getattr(person, "in_my_tree", False)

        if not in_my_tree:
            return TreeClassification.OUT_OF_TREE

        # Check if recently added
        family_tree = getattr(person, "family_tree", None)
        if family_tree:
            created_at = getattr(family_tree, "created_at", None)
            if created_at and self._is_recent(created_at, days=7):
                return TreeClassification.RECENTLY_ADDED

        return TreeClassification.IN_TREE

    def _is_recent(self, timestamp: datetime, days: int = 7) -> bool:  # noqa: PLR6301
        """Check if a timestamp is within the specified number of days."""
        if not timestamp:
            return False
        cutoff = datetime.now() - timedelta(days=days)
        return timestamp > cutoff

    def _get_status_name(self, person: Person) -> str:  # noqa: PLR6301
        """Get the status name from a person, handling enum or string."""
        status = getattr(person, "status", None)
        if status is None:
            return ""
        if hasattr(status, "name"):
            return status.name
        return str(status).upper()


# ============================================================
# Module Tests
# ============================================================


def module_tests() -> bool:
    """Run module tests for person_eligibility.py."""
    from dataclasses import dataclass as dc

    suite = TestSuite("Person Eligibility", "messaging/person_eligibility.py")
    suite.start_suite()

    # Mock Person class
    @dc
    class MockPerson:
        id: int = 1
        username: str = "test_user"
        status: str = "ACTIVE"
        profile_id: Optional[str] = "12345"
        administrator_profile_id: Optional[str] = None
        in_my_tree: bool = False
        family_tree: Optional[object] = None

    def _get_mock_person(**kwargs: Any) -> Person:
        """Create a mock Person with proper type cast."""
        return cast("Person", MockPerson(**kwargs))

    # Test 1: IneligibilityReason enum
    def test_ineligibility_enum() -> None:
        assert len(IneligibilityReason) == 9

    suite.run_test(
        "IneligibilityReason enum has 9 values",
        test_ineligibility_enum,
        expected_outcome="NONE, STATUS_ARCHIVED/BLOCKED/DEAD/DESIST, OPT_OUT_DETECTED, RATE_LIMITED, NO_CONTACT_INFO, TREE_STATUS_CHANGED",
    )

    # Test 2: TreeClassification enum
    def test_tree_classification_enum() -> None:
        assert len(TreeClassification) == 3

    suite.run_test(
        "TreeClassification enum has 3 values",
        test_tree_classification_enum,
        expected_outcome="IN_TREE, OUT_OF_TREE, RECENTLY_ADDED",
    )

    # Test 3: EligibilityResult dataclass
    result = EligibilityResult(is_eligible=True)

    def test_eligibility_result() -> None:
        assert result.is_eligible and result.reason == IneligibilityReason.NONE

    suite.run_test(
        "EligibilityResult has correct defaults",
        test_eligibility_result,
        expected_outcome="is_eligible=True, reason=NONE",
    )

    # Test 4: Checker with eligible person
    checker = PersonEligibilityChecker()
    person = _get_mock_person()
    eligibility = checker.check_eligibility(person)

    def test_eligible_person() -> None:
        assert eligibility.is_eligible

    suite.run_test(
        "Eligible person passes all checks",
        test_eligible_person,
        expected_outcome="is_eligible=True for active person with profile_id",
    )

    # Test 5: Checker with blocked status
    blocked_person = _get_mock_person(status="BLOCKED")
    blocked_result = checker.check_eligibility(blocked_person)

    def test_blocked_person() -> None:
        assert not blocked_result.is_eligible and blocked_result.reason == IneligibilityReason.STATUS_BLOCKED

    suite.run_test(
        "Blocked person is ineligible",
        test_blocked_person,
        expected_outcome="is_eligible=False, reason=STATUS_BLOCKED",
    )

    # Test 6: Checker with no contact info
    no_contact = _get_mock_person(profile_id=None, administrator_profile_id=None)
    no_contact_result = checker.check_eligibility(no_contact)

    def test_no_contact() -> None:
        assert not no_contact_result.is_eligible and no_contact_result.reason == IneligibilityReason.NO_CONTACT_INFO

    suite.run_test(
        "Person without contact info is ineligible",
        test_no_contact,
        expected_outcome="is_eligible=False, reason=NO_CONTACT_INFO",
    )

    # Test 7: Tree classification - out of tree
    out_tree_person = _get_mock_person(in_my_tree=False)
    out_tree_result = checker.check_eligibility(out_tree_person)

    def test_out_of_tree() -> None:
        assert out_tree_result.tree_classification == TreeClassification.OUT_OF_TREE

    suite.run_test(
        "Person not in tree classified as OUT_OF_TREE",
        test_out_of_tree,
        expected_outcome="tree_classification=OUT_OF_TREE",
    )

    # Test 8: RateLimitConfig defaults
    config = RateLimitConfig()

    def test_rate_limit_config() -> None:
        assert config.max_messages_per_window == 3 and config.window_hours == 72 and config.cooldown_hours == 24

    suite.run_test(
        "RateLimitConfig has correct defaults",
        test_rate_limit_config,
        expected_outcome="max=3, window=72h, cooldown=24h",
    )

    return suite.finish_suite()


# Standard test runner pattern
def run_comprehensive_tests() -> bool:
    """Standard entry point for test runner."""
    return module_tests()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
