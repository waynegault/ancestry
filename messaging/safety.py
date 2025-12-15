#!/usr/bin/env python3
"""
Safety & Ethics Module (The "Kill Switch")

Ensures all automated interactions adhere to strict safety and ethical guidelines.
Detects red flags (self-harm, hostility) and opt-out requests.

Phase 2 Enhancement (Dec 2025):
- Added CriticalAlertCategory enum for fine-grained alert classification
- Enhanced patterns per docs/specs/reply_management.md Section 3.2
- Added HIGH_VALUE category for notification-only triggers
- Added check_critical_alerts() method for integration with action7
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

# === MODULE SETUP ===
logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"  # Hostile, self-harm, threats
    OPT_OUT = "OPT_OUT"  # User requested to stop
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Ambiguous
    CRITICAL_ALERT = "CRITICAL_ALERT"  # Phase 2: Requires immediate human review
    HIGH_VALUE = "HIGH_VALUE"  # Phase 2: Valuable discovery, notify but continue


class CriticalAlertCategory(Enum):
    """Fine-grained categories for critical alerts per reply_management.md spec."""

    THREATS_HOSTILITY = "THREATS_HOSTILITY"  # Stop, cease and desist, legal threats
    SELF_HARM = "SELF_HARM"  # Suicide, self-harm indicators
    LEGAL_PRIVACY = "LEGAL_PRIVACY"  # GDPR, attorney, legal action
    HIGH_VALUE_DISCOVERY = "HIGH_VALUE_DISCOVERY"  # Family Bible, original photos
    SELF_MESSAGE = "SELF_MESSAGE"  # Attempting to send message to self


@dataclass
class SafetyCheckResult:
    status: SafetyStatus
    reason: str
    flagged_terms: list[str] = field(default_factory=list)
    category: CriticalAlertCategory | None = None  # Phase 2: Alert category


class SafetyGuard:
    """
    Guardian class that analyzes messages for safety and ethical compliance.

    Phase 2 Enhancement: Implements Critical Alert Detection per reply_management.md spec.
    Categories:
    - THREATS_HOSTILITY: Legal threats, harassment claims, hostile language
    - SELF_HARM: Suicide, distress indicators requiring immediate human review
    - LEGAL_PRIVACY: GDPR, attorney mentions, privacy violation claims
    - HIGH_VALUE_DISCOVERY: Family artifacts that warrant priority notification
    """

    # === STANDARD SAFETY PATTERNS (used by check_message method) ===
    _OPT_OUT_PATTERNS: list[str] = field(default_factory=list)  # Loaded from Data/opt_out_phrases.json

    _DANGER_PATTERNS: ClassVar[list[str]] = [
        r"\bsuicide\b",
        r"\bkill myself\b",
        r"\bend it all\b",
        r"\bpolice\b",
        r"\blawyer\b",
        r"\bsue you\b",
        r"\breport you\b",
        r"\bscam\b",
        r"\bfraud\b",
    ]

    _HOSTILITY_PATTERNS: ClassVar[list[str]] = [
        r"\bfuck\b",
        r"\bshit\b",
        r"\basshole\b",
        r"\bidiot\b",
        r"\bstupid\b",
        r"\bhate\b",
    ]

    # === PHASE 2: CRITICAL ALERT PATTERNS (per reply_management.md Section 3.2) ===

    # Category: Threats & Hostility - Stops automation, requires human review
    _CRITICAL_THREATS_PATTERNS: ClassVar[list[str]] = [
        r"stop messaging",
        r"do not contact",
        r"cease and desist",
        r"\bharassment\b",
        r"\bstalking\b",
        r"\bpolice\b",
        r"\blawyer\b",
        r"sue you",
        r"report you",
        r"\bkill\b",
        r"\bmurder\b",
        r"\bdie\b",
        r"\bhate\b",
        r"\bscam\b",
        r"\bspammer\b",
        r"\bbot\b",
    ]

    # Category: Self-Harm / Distress - HIGHEST PRIORITY, immediate human review
    _CRITICAL_SELF_HARM_PATTERNS: ClassVar[list[str]] = [
        r"\bsuicide\b",
        r"kill myself",
        r"end it all",
        r"want to die",
        r"no reason to live",
        r"hurt myself",
        r"self.harm",  # Matches self-harm, selfharm
    ]

    # Category: Legal / Privacy - Stops automation
    _CRITICAL_LEGAL_PATTERNS: ClassVar[list[str]] = [
        r"\bgdpr\b",
        r"privacy violation",
        r"data protection",
        r"\battorney\b",
        r"legal action",
        r"cease and desist",
    ]

    # Category: High Value Discovery - Notification only, does NOT stop automation
    _HIGH_VALUE_PATTERNS: ClassVar[list[str]] = [
        r"family bible",
        r"original photo",
        r"\bdaguerreotype\b",
        r"tin type",
        r"\btintype\b",
        r"marriage certificate",
        r"death certificate",
        r"birth certificate",
        r"\bwill\b",  # Context-sensitive, may need human review
        r"\bdeed\b",
        r"\bdiary\b",
        r"\bjournal\b",
        r"\bletters\b",
        r"old photographs",
        r"family documents",
    ]

    def __init__(self) -> None:
        self._load_opt_out_patterns()
        self._compile_patterns()

    def _load_opt_out_patterns(self) -> None:
        """Load opt-out patterns from Data/opt_out_phrases.json."""
        try:
            # Resolve path relative to project root
            # This file is in messaging/safety.py, so root is two levels up
            base_path = Path(__file__).resolve().parent.parent
            json_path = base_path / "Data" / "opt_out_phrases.json"

            if json_path.exists():
                with Path(json_path).open(encoding="utf-8") as f:
                    self._OPT_OUT_PATTERNS = json.load(f)
            else:
                logger.warning(f"Opt-out phrases file not found at {json_path}. Using fallback.")
                self._OPT_OUT_PATTERNS = [r"\bstop messaging\b"]
        except Exception as e:
            logger.error(f"Failed to load opt-out patterns: {e}")
            self._OPT_OUT_PATTERNS = [r"\bstop messaging\b"]

    def _compile_patterns(self) -> None:
        # Legacy patterns
        self.opt_out_regex = [re.compile(p, re.IGNORECASE) for p in self._OPT_OUT_PATTERNS]
        self.danger_regex = [re.compile(p, re.IGNORECASE) for p in self._DANGER_PATTERNS]
        self.hostility_regex = [re.compile(p, re.IGNORECASE) for p in self._HOSTILITY_PATTERNS]

        # Phase 2: Critical Alert patterns
        self.critical_threats_regex = [re.compile(p, re.IGNORECASE) for p in self._CRITICAL_THREATS_PATTERNS]
        self.critical_self_harm_regex = [re.compile(p, re.IGNORECASE) for p in self._CRITICAL_SELF_HARM_PATTERNS]
        self.critical_legal_regex = [re.compile(p, re.IGNORECASE) for p in self._CRITICAL_LEGAL_PATTERNS]
        self.high_value_regex = [re.compile(p, re.IGNORECASE) for p in self._HIGH_VALUE_PATTERNS]

    def check_critical_alerts(self, message_text: str) -> SafetyCheckResult:
        """
        Phase 2: Check for Critical Alerts that require human review.

        This method should be called BEFORE AI classification in action7.
        If a critical alert is detected, automation should STOP for this conversation.

        Priority order:
        1. Self-harm (highest) - Immediate human intervention needed
        2. Threats/Hostility - Stop automation, flag for review
        3. Legal/Privacy - Stop automation, flag for review
        4. High Value - Continue automation, but notify user (priority flag)

        Args:
            message_text: The content of the inbound message.

        Returns:
            SafetyCheckResult with category for fine-grained handling.
        """
        if not message_text:
            return SafetyCheckResult(SafetyStatus.SAFE, "Empty message", [])

        # 1. SELF-HARM (Highest Priority)
        flagged_self_harm = self._find_matches(message_text, self.critical_self_harm_regex)
        if flagged_self_harm:
            logger.critical(f"🚨 CRITICAL ALERT: Self-harm indicators detected: {flagged_self_harm}")
            return SafetyCheckResult(
                status=SafetyStatus.CRITICAL_ALERT,
                reason="Self-harm/distress indicators detected - REQUIRES IMMEDIATE HUMAN REVIEW",
                flagged_terms=flagged_self_harm,
                category=CriticalAlertCategory.SELF_HARM,
            )

        # 2. THREATS & HOSTILITY
        flagged_threats = self._find_matches(message_text, self.critical_threats_regex)
        if flagged_threats:
            logger.error(f"⚠️ Critical Alert: Threat/hostility detected: {flagged_threats}")
            return SafetyCheckResult(
                status=SafetyStatus.CRITICAL_ALERT,
                reason="Threats or hostile language detected",
                flagged_terms=flagged_threats,
                category=CriticalAlertCategory.THREATS_HOSTILITY,
            )

        # 3. LEGAL / PRIVACY
        flagged_legal = self._find_matches(message_text, self.critical_legal_regex)
        if flagged_legal:
            logger.warning(f"⚠️ Critical Alert: Legal/privacy concern detected: {flagged_legal}")
            return SafetyCheckResult(
                status=SafetyStatus.CRITICAL_ALERT,
                reason="Legal or privacy-related concern detected",
                flagged_terms=flagged_legal,
                category=CriticalAlertCategory.LEGAL_PRIVACY,
            )

        # 4. HIGH VALUE DISCOVERY (Notification only - does NOT stop automation)
        flagged_high_value = self._find_matches(message_text, self.high_value_regex)
        if flagged_high_value:
            logger.info(f"📚 High-value discovery mentioned: {flagged_high_value}")
            return SafetyCheckResult(
                status=SafetyStatus.HIGH_VALUE,
                reason="High-value genealogical artifact mentioned - flag for priority follow-up",
                flagged_terms=flagged_high_value,
                category=CriticalAlertCategory.HIGH_VALUE_DISCOVERY,
            )

        return SafetyCheckResult(SafetyStatus.SAFE, "No critical alerts", [])

    @staticmethod
    def check_self_message(
        sender_profile_id: str | None,
        sender_uuid: str | None,
        recipient_profile_id: str | None,
        recipient_uuid: str | None,
    ) -> SafetyCheckResult:
        """
        Check if a message is being sent to self.

        This prevents the automation from sending messages to the owner's own profile.

        Args:
            sender_profile_id: The profile ID of the sender (owner).
            sender_uuid: The UUID of the sender (owner).
            recipient_profile_id: The profile ID of the recipient.
            recipient_uuid: The UUID of the recipient.

        Returns:
            SafetyCheckResult with CRITICAL_ALERT if self-message detected.
        """
        # Check by profile_id
        if (
            sender_profile_id
            and recipient_profile_id
            and sender_profile_id.strip().lower() == recipient_profile_id.strip().lower()
        ):
            logger.critical(
                f"🚨 SELF_MESSAGE: Blocked message to self (profile_id={sender_profile_id})"
            )
            return SafetyCheckResult(
                status=SafetyStatus.CRITICAL_ALERT,
                reason="Attempted to send message to self (same profile_id)",
                flagged_terms=[f"profile_id={sender_profile_id}"],
                category=CriticalAlertCategory.SELF_MESSAGE,
            )

        # Check by UUID
        if (
            sender_uuid
            and recipient_uuid
            and sender_uuid.strip().upper() == recipient_uuid.strip().upper()
        ):
            logger.critical(f"🚨 SELF_MESSAGE: Blocked message to self (uuid={sender_uuid})")
            return SafetyCheckResult(
                status=SafetyStatus.CRITICAL_ALERT,
                reason="Attempted to send message to self (same UUID)",
                flagged_terms=[f"uuid={sender_uuid}"],
                category=CriticalAlertCategory.SELF_MESSAGE,
            )

        return SafetyCheckResult(SafetyStatus.SAFE, "Not a self-message", [])

    def check_message(self, message_text: str) -> SafetyCheckResult:
        """
        Analyzes a message for safety violations (legacy method).

        Args:
            message_text: The content of the inbound message.

        Returns:
            SafetyCheckResult object.
        """
        if not message_text:
            return SafetyCheckResult(SafetyStatus.SAFE, "Empty message", [])

        # 1. Check for Danger/Legal Threats (Highest Priority)
        flagged_danger = self._find_matches(message_text, self.danger_regex)
        if flagged_danger:
            return SafetyCheckResult(SafetyStatus.UNSAFE, "Danger/Legal threat detected", flagged_danger)

        # 2. Check for Opt-Out
        flagged_opt_out = self._find_matches(message_text, self.opt_out_regex)
        if flagged_opt_out:
            return SafetyCheckResult(SafetyStatus.OPT_OUT, "User requested opt-out", flagged_opt_out)

        # 3. Check for Hostility
        flagged_hostility = self._find_matches(message_text, self.hostility_regex)
        if flagged_hostility:
            return SafetyCheckResult(SafetyStatus.UNSAFE, "Hostile language detected", flagged_hostility)

        return SafetyCheckResult(SafetyStatus.SAFE, "No flags detected", [])

    @staticmethod
    def _find_matches(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
        matches: list[str] = []
        for pattern in patterns:
            found: list[str] = pattern.findall(text)
            if found:
                matches.extend(found)
        return list(set(matches))  # Deduplicate


# === TEST SUITE ===
def safety_module_tests() -> bool:
    """Run tests for the SafetyGuard module."""
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Safety Module", __name__)
        suite.start_suite()

        guard = SafetyGuard()

        # === LEGACY check_message TESTS ===
        def test_safe_message():
            msg = "Hello, I would love to share information about our family."
            result = guard.check_message(msg)
            assert result.status == SafetyStatus.SAFE
            assert len(result.flagged_terms) == 0

        def test_opt_out():
            msg = "Please stop messaging me. I am not interested."
            result = guard.check_message(msg)
            assert result.status == SafetyStatus.OPT_OUT
            # Updated to check for full phrases instead of single words
            flagged = [t.lower() for t in result.flagged_terms]
            assert any("stop messaging me" in t for t in flagged) or any("not interested" in t for t in flagged)

        def test_false_positive_stop():
            """Ensure 'stop' used in normal context doesn't trigger opt-out."""
            msg = "We made a stop to visit us in the summer."
            result = guard.check_message(msg)
            assert result.status == SafetyStatus.SAFE

        def test_danger():
            msg = "This is a scam, I will call the police."
            result = guard.check_message(msg)
            assert result.status == SafetyStatus.UNSAFE
            assert "police" in [t.lower() for t in result.flagged_terms]

        def test_hostility():
            msg = "You are an idiot."
            result = guard.check_message(msg)
            assert result.status == SafetyStatus.UNSAFE
            assert "idiot" in [t.lower() for t in result.flagged_terms]

        suite.run_test("Safe Message", test_safe_message, "Should return SAFE")
        suite.run_test("Opt Out", test_opt_out, "Should return OPT_OUT")
        suite.run_test("False Positive Stop", test_false_positive_stop, "Should return SAFE")
        suite.run_test("Danger Detection", test_danger, "Should return UNSAFE")
        suite.run_test("Hostility Detection", test_hostility, "Should return UNSAFE")

        # === PHASE 2: CRITICAL ALERT TESTS ===
        def test_critical_self_harm():
            """Self-harm is highest priority critical alert."""
            msg = "I've been thinking about suicide lately. Life seems pointless."
            result = guard.check_critical_alerts(msg)
            assert result.status == SafetyStatus.CRITICAL_ALERT
            assert result.category == CriticalAlertCategory.SELF_HARM
            assert "suicide" in [t.lower() for t in result.flagged_terms]

        def test_critical_threats():
            """Threats/hostility triggers critical alert."""
            msg = "Stop messaging me or I will report you to the police and sue you."
            result = guard.check_critical_alerts(msg)
            assert result.status == SafetyStatus.CRITICAL_ALERT
            assert result.category == CriticalAlertCategory.THREATS_HOSTILITY

        def test_critical_legal():
            """Legal/privacy triggers critical alert."""
            msg = "Under GDPR I demand you delete all my data. My attorney will be in contact."
            result = guard.check_critical_alerts(msg)
            assert result.status == SafetyStatus.CRITICAL_ALERT
            assert result.category == CriticalAlertCategory.LEGAL_PRIVACY

        def test_high_value_discovery():
            """High value items trigger notification, NOT stop."""
            msg = "I found my great-grandmother's family bible with all the births recorded!"
            result = guard.check_critical_alerts(msg)
            assert result.status == SafetyStatus.HIGH_VALUE  # Not CRITICAL_ALERT
            assert result.category == CriticalAlertCategory.HIGH_VALUE_DISCOVERY

        def test_critical_safe_message():
            """Normal genealogical message passes critical alert check."""
            msg = "I'd love to share what I know about the Simpson family in Aberdeen."
            result = guard.check_critical_alerts(msg)
            assert result.status == SafetyStatus.SAFE
            assert result.category is None

        def test_critical_priority_order():
            """Self-harm takes priority over threats when both present."""
            msg = "I want to die. I hate you and will sue you."
            result = guard.check_critical_alerts(msg)
            assert result.status == SafetyStatus.CRITICAL_ALERT
            assert result.category == CriticalAlertCategory.SELF_HARM  # Self-harm > threats

        suite.run_test("Critical: Self-harm", test_critical_self_harm, "Self-harm -> CRITICAL_ALERT")
        suite.run_test("Critical: Threats", test_critical_threats, "Threats -> CRITICAL_ALERT")
        suite.run_test("Critical: Legal", test_critical_legal, "Legal -> CRITICAL_ALERT")
        suite.run_test("Critical: High Value", test_high_value_discovery, "High value -> HIGH_VALUE (not stop)")
        suite.run_test("Critical: Safe", test_critical_safe_message, "Normal message -> SAFE")
        suite.run_test("Critical: Priority", test_critical_priority_order, "Self-harm priority over threats")

        # === SELF-MESSAGE DETECTION TESTS ===
        def test_self_message_by_profile_id():
            """Detect self-message by matching profile_id."""
            result = guard.check_self_message(
                sender_profile_id="12345",
                sender_uuid=None,
                recipient_profile_id="12345",
                recipient_uuid=None,
            )
            assert result.status == SafetyStatus.CRITICAL_ALERT
            assert result.category == CriticalAlertCategory.SELF_MESSAGE

        def test_self_message_by_uuid():
            """Detect self-message by matching UUID (case-insensitive)."""
            result = guard.check_self_message(
                sender_profile_id=None,
                sender_uuid="ABC123",
                recipient_profile_id=None,
                recipient_uuid="abc123",
            )
            assert result.status == SafetyStatus.CRITICAL_ALERT
            assert result.category == CriticalAlertCategory.SELF_MESSAGE

        def test_self_message_different_users():
            """Different profile_id/UUID passes self-message check."""
            result = guard.check_self_message(
                sender_profile_id="12345",
                sender_uuid="ABC123",
                recipient_profile_id="67890",
                recipient_uuid="DEF456",
            )
            assert result.status == SafetyStatus.SAFE
            assert result.category is None

        def test_self_message_none_values():
            """None values should not match as self-message."""
            result = guard.check_self_message(
                sender_profile_id=None,
                sender_uuid=None,
                recipient_profile_id=None,
                recipient_uuid=None,
            )
            assert result.status == SafetyStatus.SAFE

        suite.run_test(
            "Self-Message: profile_id match",
            test_self_message_by_profile_id,
            "Same profile_id -> CRITICAL_ALERT",
        )
        suite.run_test(
            "Self-Message: UUID match",
            test_self_message_by_uuid,
            "Same UUID (case-insensitive) -> CRITICAL_ALERT",
        )
        suite.run_test(
            "Self-Message: Different users",
            test_self_message_different_users,
            "Different IDs -> SAFE",
        )
        suite.run_test(
            "Self-Message: None values",
            test_self_message_none_values,
            "None values -> SAFE",
        )

        return suite.finish_suite()


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(safety_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
