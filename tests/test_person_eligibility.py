"""Regression tests for the person eligibility opt-out check.

The opt-out branch of PersonEligibilityChecker used to import
``messaging.opt_out_detection``, a module that does not exist. The resulting
ImportError was caught by the surrounding ``except Exception`` and the check
failed open, so an opted-out person was reported eligible and could be
messaged. These tests pin the check to the real detector in
``core.opt_out_detection``.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from messaging.person_eligibility import (
    IneligibilityReason,
    PersonEligibilityChecker,
    PersonEligibilityContext,
)


@dataclass
class _FakePerson:
    """Minimal stand-in for a Person row; only the id is used by the opt-out check."""

    id: int


class TestPersonEligibilityOptOut(unittest.TestCase):
    def test_opted_out_person_is_not_eligible(self) -> None:
        checker = PersonEligibilityChecker(db_session=MagicMock())
        person: Any = _FakePerson(4242)

        with patch(
            "core.opt_out_detection.OptOutDetector.validate_can_send",
            return_value=(False, "asked to stop"),
        ):
            result = checker._check_opt_out(person, PersonEligibilityContext())

        self.assertFalse(result.is_eligible)
        self.assertIs(result.reason, IneligibilityReason.OPT_OUT_DETECTED)
        self.assertIn("asked to stop", result.reason_detail)

    def test_cleared_person_is_eligible(self) -> None:
        checker = PersonEligibilityChecker(db_session=MagicMock())
        person: Any = _FakePerson(4243)

        with patch(
            "core.opt_out_detection.OptOutDetector.validate_can_send",
            return_value=(True, ""),
        ):
            result = checker._check_opt_out(person, PersonEligibilityContext())

        self.assertTrue(result.is_eligible)
        self.assertIs(result.reason, IneligibilityReason.NONE)


if __name__ == "__main__":
    unittest.main()
