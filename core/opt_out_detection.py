#!/usr/bin/env python3

"""
Opt-Out Detection and Safeguards

Sprint 4: Enhanced opt-out detection and prevention of messages to opted-out users.
Provides multi-layer safeguards against sending unwanted messages.

Key Features:
- DESIST status detection from messages
- Database-level opt-out tracking
- Pre-send validation
- Conversation history analysis
- Automatic status updates
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)


# === DATA CLASSES ===


@dataclass()
class OptOutIndicator:
    """An indicator that suggests opt-out intent."""

    pattern: str  # Text pattern or keyword
    confidence: float  # 0.0 to 1.0
    category: str  # "explicit", "implicit", "aggressive"
    example: str  # Example of matching text


@dataclass()
class OptOutAnalysis:
    """Result of analyzing a message for opt-out indicators."""

    is_opt_out: bool
    confidence: float  # 0.0 to 1.0
    indicators_found: list[str]
    suggested_action: str  # "block", "flag", "proceed"
    raw_text: str


@dataclass
class PersonOptOutStatus:
    """Current opt-out status for a person."""

    person_id: int
    is_opted_out: bool
    opt_out_reason: Optional[str]
    opt_out_date: Optional[datetime]
    can_contact: bool
    blocklist_reason: Optional[str] = None


# === OPT-OUT PATTERNS ===


# Explicit opt-out phrases (high confidence)
EXPLICIT_OPT_OUT_PATTERNS = [
    OptOutIndicator(
        pattern=r"\bstop\s+contact(ing)?\b",
        confidence=0.95,
        category="explicit",
        example="Please stop contacting me",
    ),
    OptOutIndicator(
        pattern=r"\bdo\s+not\s+(contact|message|email|write)\b",
        confidence=0.95,
        category="explicit",
        example="Do not contact me again",
    ),
    OptOutIndicator(
        pattern=r"\bremove\s+(me|my\s+(email|name))\b",
        confidence=0.90,
        category="explicit",
        example="Remove me from your list",
    ),
    OptOutIndicator(
        pattern=r"\bunsubscribe\b",
        confidence=0.90,
        category="explicit",
        example="Please unsubscribe me",
    ),
    OptOutIndicator(
        pattern=r"\bopt\s*(-|\s)?out\b",
        confidence=0.95,
        category="explicit",
        example="I want to opt out",
    ),
    OptOutIndicator(
        pattern=r"\bleave\s+me\s+alone\b",
        confidence=0.95,
        category="explicit",
        example="Leave me alone",
    ),
    OptOutIndicator(
        pattern=r"\bno\s+more\s+(messages?|emails?|contact)\b",
        confidence=0.90,
        category="explicit",
        example="No more messages please",
    ),
    OptOutIndicator(
        pattern=r"\bstop\s+messaging\s+me\b",
        confidence=0.95,
        category="explicit",
        example="Please stop messaging me",
    ),
]

# Implicit opt-out phrases (medium confidence)
IMPLICIT_OPT_OUT_PATTERNS = [
    OptOutIndicator(
        pattern=r"\bnot\s+interested\b",
        confidence=0.70,
        category="implicit",
        example="I'm not interested",
    ),
    OptOutIndicator(
        pattern=r"\bplease\s+don'?t\b",
        confidence=0.65,
        category="implicit",
        example="Please don't message me",
    ),
    OptOutIndicator(
        pattern=r"\bwrong\s+(person|email|address)\b",
        confidence=0.80,
        category="implicit",
        example="You have the wrong person",
    ),
    OptOutIndicator(
        pattern=r"\bnot\s+(related|family)\b",
        confidence=0.75,
        category="implicit",
        example="We're not related",
    ),
]

# Aggressive/hostile patterns (block immediately)
AGGRESSIVE_PATTERNS = [
    OptOutIndicator(
        pattern=r"\b(harassment|harassing)\b",
        confidence=0.98,
        category="aggressive",
        example="This is harassment",
    ),
    OptOutIndicator(
        pattern=r"\bspam(ming)?\b",
        confidence=0.85,
        category="aggressive",
        example="Stop spamming me",
    ),
    OptOutIndicator(
        pattern=r"\breport(ing|ed)?\s+(you|this)\b",
        confidence=0.90,
        category="aggressive",
        example="I will report you",
    ),
    OptOutIndicator(
        pattern=r"\blegal\s+action\b",
        confidence=0.95,
        category="aggressive",
        example="I will take legal action",
    ),
    OptOutIndicator(
        pattern=r"\bblock(ed|ing)?\s+(you|your)\b",
        confidence=0.85,
        category="aggressive",
        example="I'm blocking you",
    ),
]


# === OPT-OUT DETECTOR ===


class OptOutDetector:
    """
    Detects opt-out indicators in messages and conversation history.

    Provides multi-layer detection:
    1. Pattern matching against explicit/implicit phrases
    2. Sentiment analysis integration
    3. Conversation history analysis
    4. Database status checking
    """

    # Thresholds
    OPT_OUT_THRESHOLD = 0.75  # Minimum confidence to consider opt-out
    BLOCK_THRESHOLD = 0.90  # Confidence threshold for immediate block

    def __init__(self, db_session: Optional[DbSession] = None) -> None:
        """Initialize the opt-out detector."""
        self.db_session = db_session
        self._explicit_patterns = EXPLICIT_OPT_OUT_PATTERNS
        self._implicit_patterns = IMPLICIT_OPT_OUT_PATTERNS
        self._aggressive_patterns = AGGRESSIVE_PATTERNS

    def analyze_message(self, message_text: str) -> OptOutAnalysis:
        """
        Analyze a message for opt-out indicators.

        Args:
            message_text: The message text to analyze

        Returns:
            OptOutAnalysis with detection results
        """
        if not message_text:
            return OptOutAnalysis(
                is_opt_out=False,
                confidence=0.0,
                indicators_found=[],
                suggested_action="proceed",
                raw_text="",
            )

        text_lower = message_text.lower()
        indicators_found: list[str] = []
        max_confidence = 0.0
        has_aggressive = False

        # Check all patterns
        for pattern in self._explicit_patterns + self._implicit_patterns + self._aggressive_patterns:
            if re.search(pattern.pattern, text_lower, re.IGNORECASE):
                indicators_found.append(f"{pattern.category}: {pattern.example}")
                max_confidence = max(max_confidence, pattern.confidence)
                if pattern.category == "aggressive":
                    has_aggressive = True

        # Determine action
        if has_aggressive or max_confidence >= self.BLOCK_THRESHOLD:
            suggested_action = "block"
        elif max_confidence >= self.OPT_OUT_THRESHOLD:
            suggested_action = "flag"
        else:
            suggested_action = "proceed"

        return OptOutAnalysis(
            is_opt_out=max_confidence >= self.OPT_OUT_THRESHOLD,
            confidence=max_confidence,
            indicators_found=indicators_found,
            suggested_action=suggested_action,
            raw_text=message_text[:200],  # Truncate for logging
        )

    def check_person_status(self, person_id: int) -> PersonOptOutStatus:
        """
        Check opt-out status for a person in the database.

        Args:
            person_id: ID of the person to check

        Returns:
            PersonOptOutStatus with current status
        """
        if not self.db_session:
            return PersonOptOutStatus(
                person_id=person_id,
                is_opted_out=False,
                opt_out_reason=None,
                opt_out_date=None,
                can_contact=True,
            )

        try:
            from core.database import Person, PersonStatusEnum

            person = self.db_session.query(Person).filter(Person.id == person_id).first()
            if not person:
                return PersonOptOutStatus(
                    person_id=person_id,
                    is_opted_out=False,
                    opt_out_reason="Person not found",
                    opt_out_date=None,
                    can_contact=False,  # Can't contact if not found
                )

            is_desist = person.status == PersonStatusEnum.DESIST
            is_contactable = person.contactable if hasattr(person, "contactable") else True

            return PersonOptOutStatus(
                person_id=person_id,
                is_opted_out=is_desist,
                opt_out_reason="DESIST status" if is_desist else None,
                opt_out_date=None,  # Would need to track this
                can_contact=is_contactable and not is_desist,
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to check person status: {e}")
            return PersonOptOutStatus(
                person_id=person_id,
                is_opted_out=False,
                opt_out_reason=f"Database error: {e}",
                opt_out_date=None,
                can_contact=False,  # Err on side of caution
            )

    def check_conversation_history(self, person_id: int) -> bool:
        """
        Check conversation history for past opt-out indicators.

        Args:
            person_id: ID of the person

        Returns:
            True if opt-out detected in history, False otherwise
        """
        if not self.db_session:
            return False

        try:
            from core.database import ConversationLog

            # Get recent inbound messages
            messages = (
                self.db_session.query(ConversationLog)
                .filter(
                    and_(
                        ConversationLog.person_id == person_id,
                        ConversationLog.direction == "IN",
                    )
                )
                .order_by(ConversationLog.timestamp.desc())
                .limit(10)
                .all()
            )

            for msg in messages:
                analysis = self.analyze_message(msg.content or "")
                if analysis.is_opt_out:
                    logger.info(f"Found opt-out indicator in conversation history for person {person_id}")
                    return True

            return False

        except SQLAlchemyError as e:
            logger.error(f"Failed to check conversation history: {e}")
            return False

    def validate_can_send(self, person_id: int) -> tuple[bool, str]:
        """
        Validate whether a message can be sent to a person.

        Args:
            person_id: ID of the person

        Returns:
            Tuple of (can_send, reason)
        """
        # Check database status
        status = self.check_person_status(person_id)
        if not status.can_contact:
            reason = status.opt_out_reason or "Person cannot be contacted"
            return False, reason

        # Check conversation history
        if self.check_conversation_history(person_id):
            return False, "Opt-out detected in conversation history"

        return True, "OK"

    def mark_opted_out(self, person_id: int, reason: str) -> bool:
        """
        Mark a person as opted out (DESIST status).

        Args:
            person_id: ID of the person
            reason: Reason for opt-out

        Returns:
            True if successful, False otherwise
        """
        if not self.db_session:
            logger.warning("No database session available for opt-out marking")
            return False

        try:
            from core.database import Person, PersonStatusEnum

            person = self.db_session.query(Person).filter(Person.id == person_id).first()
            if not person:
                logger.warning(f"Person {person_id} not found for opt-out marking")
                return False

            person.status = PersonStatusEnum.DESIST
            self.db_session.commit()

            logger.info(f"Marked person {person_id} as DESIST: {reason}")
            return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to mark person as opted out: {e}")
            self.db_session.rollback()
            return False


# === CONVENIENCE FUNCTIONS ===


def detect_opt_out(message_text: str) -> OptOutAnalysis:
    """
    Quick opt-out detection without database.

    Args:
        message_text: Message to analyze

    Returns:
        OptOutAnalysis result
    """
    detector = OptOutDetector()
    return detector.analyze_message(message_text)


def can_send_message(person_id: int, db_session: DbSession) -> tuple[bool, str]:
    """
    Check if a message can be sent to a person.

    Args:
        person_id: ID of the person
        db_session: Database session

    Returns:
        Tuple of (can_send, reason)
    """
    detector = OptOutDetector(db_session)
    return detector.validate_can_send(person_id)


# === MODULE TESTS ===


def module_tests() -> bool:
    """Run module-specific tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Opt-Out Detection & Safeguards", "core/opt_out_detection.py")
    suite.start_suite()

    # Test 1: Explicit opt-out detection
    def test_explicit_opt_out() -> None:
        detector = OptOutDetector()
        analysis = detector.analyze_message("Please stop contacting me")
        assert analysis.is_opt_out, "Should detect explicit opt-out"
        assert analysis.confidence >= 0.9, "Should have high confidence"
        assert analysis.suggested_action == "block", "Should suggest blocking"

    suite.run_test(
        "Explicit opt-out detection",
        test_explicit_opt_out,
        test_summary="Detects explicit opt-out phrases",
        functions_tested="OptOutDetector.analyze_message",
        method_description="Analyze message with 'stop contacting'",
    )

    # Test 2: Implicit opt-out detection
    def test_implicit_opt_out() -> None:
        detector = OptOutDetector()
        analysis = detector.analyze_message("I'm not interested in this")
        assert analysis.is_opt_out is False or analysis.confidence < 0.9, (
            "Implicit opt-out should have lower confidence"
        )

    suite.run_test(
        "Implicit opt-out detection",
        test_implicit_opt_out,
        test_summary="Detects implicit opt-out phrases with lower confidence",
        functions_tested="OptOutDetector.analyze_message",
        method_description="Analyze message with 'not interested'",
    )

    # Test 3: Aggressive pattern detection
    def test_aggressive_detection() -> None:
        detector = OptOutDetector()
        analysis = detector.analyze_message("This is harassment, I will report you")
        assert analysis.is_opt_out, "Should detect aggressive opt-out"
        assert analysis.suggested_action == "block", "Should suggest immediate block"

    suite.run_test(
        "Aggressive pattern detection",
        test_aggressive_detection,
        test_summary="Detects aggressive language and suggests block",
        functions_tested="OptOutDetector.analyze_message",
        method_description="Analyze message with 'harassment'",
    )

    # Test 4: Normal message (no opt-out)
    def test_normal_message() -> None:
        detector = OptOutDetector()
        analysis = detector.analyze_message("Thanks for reaching out! I'd love to learn more about our connection.")
        assert not analysis.is_opt_out, "Should not detect opt-out in friendly message"
        assert analysis.suggested_action == "proceed", "Should allow proceeding"

    suite.run_test(
        "Normal message detection",
        test_normal_message,
        test_summary="Allows friendly messages through",
        functions_tested="OptOutDetector.analyze_message",
        method_description="Analyze friendly response message",
    )

    # Test 5: Empty message handling
    def test_empty_message() -> None:
        detector = OptOutDetector()
        analysis = detector.analyze_message("")
        assert not analysis.is_opt_out, "Empty message should not be opt-out"
        assert analysis.confidence == 0.0, "Confidence should be 0"

    suite.run_test(
        "Empty message handling",
        test_empty_message,
        test_summary="Handles empty messages gracefully",
        functions_tested="OptOutDetector.analyze_message",
        method_description="Analyze empty string",
    )

    # Test 6: PersonOptOutStatus dataclass
    def test_status_dataclass() -> None:
        status = PersonOptOutStatus(
            person_id=123,
            is_opted_out=True,
            opt_out_reason="DESIST status",
            opt_out_date=datetime.now(timezone.utc),
            can_contact=False,
        )
        assert status.person_id == 123, "person_id should be 123"
        assert not status.can_contact, "can_contact should be False"

    suite.run_test(
        "PersonOptOutStatus dataclass",
        test_status_dataclass,
        test_summary="PersonOptOutStatus stores values correctly",
        functions_tested="PersonOptOutStatus dataclass",
        method_description="Create status and verify fields",
    )

    # Test 7: OptOutAnalysis dataclass
    def test_analysis_dataclass() -> None:
        analysis = OptOutAnalysis(
            is_opt_out=True,
            confidence=0.95,
            indicators_found=["explicit: Leave me alone"],
            suggested_action="block",
            raw_text="Leave me alone!",
        )
        assert analysis.is_opt_out, "is_opt_out should be True"
        assert len(analysis.indicators_found) == 1, "Should have one indicator"

    suite.run_test(
        "OptOutAnalysis dataclass",
        test_analysis_dataclass,
        test_summary="OptOutAnalysis stores values correctly",
        functions_tested="OptOutAnalysis dataclass",
        method_description="Create analysis and verify fields",
    )

    # Test 8: Multiple patterns in one message
    def test_multiple_patterns() -> None:
        detector = OptOutDetector()
        analysis = detector.analyze_message("Stop contacting me! This is spam and I will report you!")
        assert analysis.is_opt_out, "Should detect opt-out"
        assert len(analysis.indicators_found) >= 2, "Should find multiple indicators"

    suite.run_test(
        "Multiple pattern detection",
        test_multiple_patterns,
        test_summary="Detects multiple opt-out patterns in one message",
        functions_tested="OptOutDetector.analyze_message",
        method_description="Analyze message with multiple opt-out phrases",
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
