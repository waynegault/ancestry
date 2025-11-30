#!/usr/bin/env python3
"""
Safety & Ethics Module (The "Kill Switch")

Ensures all automated interactions adhere to strict safety and ethical guidelines.
Detects red flags (self-harm, hostility) and opt-out requests.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

# === MODULE SETUP ===
logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"  # Hostile, self-harm, threats
    OPT_OUT = "OPT_OUT"  # User requested to stop
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Ambiguous


@dataclass
class SafetyCheckResult:
    status: SafetyStatus
    reason: str
    flagged_terms: list[str] = field(default_factory=list)


class SafetyGuard:
    """
    Guardian class that analyzes messages for safety and ethical compliance.
    """

    # Regex Patterns
    _OPT_OUT_PATTERNS: ClassVar[list[str]] = [
        r"\bstop\b",
        r"\bunsubscribe\b",
        r"\bremove me\b",
        r"\bnot interested\b",
        r"\bdon't message\b",
        r"\bdo not message\b",
        r"\bspam\b",
        r"\bharassment\b",
        r"\bleave me alone\b",
    ]

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

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.opt_out_regex = [re.compile(p, re.IGNORECASE) for p in self._OPT_OUT_PATTERNS]
        self.danger_regex = [re.compile(p, re.IGNORECASE) for p in self._DANGER_PATTERNS]
        self.hostility_regex = [re.compile(p, re.IGNORECASE) for p in self._HOSTILITY_PATTERNS]

    def check_message(self, message_text: str) -> SafetyCheckResult:
        """
        Analyzes a message for safety violations.

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
    def _find_matches(text: str, patterns: list[re.Pattern]) -> list[str]:
        matches = []
        for pattern in patterns:
            found = pattern.findall(text)
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

        def test_safe_message():
            msg = "Hello, I would love to share information about our family."
            result = guard.check_message(msg)
            assert result.status == SafetyStatus.SAFE
            assert len(result.flagged_terms) == 0

        def test_opt_out():
            msg = "Please stop messaging me. I am not interested."
            result = guard.check_message(msg)
            assert result.status == SafetyStatus.OPT_OUT
            assert "stop" in [t.lower() for t in result.flagged_terms] or "not interested" in [
                t.lower() for t in result.flagged_terms
            ]

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
        suite.run_test("Danger Detection", test_danger, "Should return UNSAFE")
        suite.run_test("Hostility Detection", test_hostility, "Should return UNSAFE")

        return suite.finish_suite()


if __name__ == "__main__":
    import sys

    # Simple runner if executed directly
    if safety_module_tests():
        sys.exit(0)
    else:
        sys.exit(1)
