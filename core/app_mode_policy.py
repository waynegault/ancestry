"""App mode policy helpers.

This module centralizes the operational mode behavior for outbound actions.
Supported modes are:
- dry_run: do everything but never send real messages
- testing: allow real sends but only to a small allowlist
- production: allow full sending

The goal is to keep user-facing configuration simple (APP_MODE), while still
allowing optional TEST_* overrides when present.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from config import config_schema


def _normalize_identifier(value: Optional[str]) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


DEFAULT_TESTING_ALLOWED_IDENTIFIERS: tuple[str, ...] = (
    # Human-friendly display names normalize to these.
    "francesmchardy",
    "francesmilne",
)


@dataclass(frozen=True, slots=True)
class OutboundPolicyDecision:
    allowed: bool
    reason: str = ""


def _iter_testing_allowed_identifiers() -> Iterable[str]:
    # 1) Hard-coded allowlist requested by user (Frances McHardy/Milne)
    yield from DEFAULT_TESTING_ALLOWED_IDENTIFIERS

    # 2) Optional env-backed testing username override (kept for compatibility)
    testing_username = getattr(config_schema, "testing_username", None)
    if testing_username:
        yield _normalize_identifier(testing_username)


def _decide_testing_mode(
    *,
    testing_profile_id: Optional[str],
    person_profile_id: Any,
    normalized_username: str,
) -> OutboundPolicyDecision:
    allowed = False
    reason = ""

    if testing_profile_id and person_profile_id and str(person_profile_id) == str(testing_profile_id):
        allowed = True
    else:
        allowed_identifiers = {_normalize_identifier(item) for item in _iter_testing_allowed_identifiers() if item}
        if normalized_username and normalized_username in allowed_identifiers:
            allowed = True
        elif testing_profile_id:
            reason = f"skipped (testing_mode_filter: not {testing_profile_id})"
        else:
            reason = "skipped (testing_mode_filter: not allowlisted)"

    return OutboundPolicyDecision(allowed, reason)


def _decide_production_mode(
    *,
    testing_profile_id: Optional[str],
    person_profile_id: Any,
) -> OutboundPolicyDecision:
    allowed = True
    reason = ""
    if testing_profile_id and person_profile_id and str(person_profile_id) == str(testing_profile_id):
        allowed = False
        reason = f"skipped (production_mode_filter: is {testing_profile_id})"
    return OutboundPolicyDecision(allowed, reason)


def should_allow_outbound_to_person(
    person: Any,
    *,
    app_mode: Optional[str] = None,
) -> OutboundPolicyDecision:
    """Return whether outbound actions should proceed for this person.

    This is designed for use by Action 8/9 and API send paths.

    Notes:
    - In dry_run, we allow the action to proceed (sending is simulated elsewhere).
    - In testing, we allow only the configured test profile (if present) or the
      allowlisted usernames (Frances McHardy/Milne).
    - In production, we optionally *block* sends to the configured test profile.
    """

    mode = (app_mode or getattr(config_schema, "app_mode", "dry_run") or "dry_run").strip().lower()

    # Profile ID checks
    testing_profile_id = getattr(config_schema, "testing_profile_id", None)
    person_profile_id = getattr(person, "profile_id", None)

    # Username allowlist checks
    person_username = getattr(person, "username", None)
    normalized_username = _normalize_identifier(person_username)

    if mode == "testing":
        decision = _decide_testing_mode(
            testing_profile_id=testing_profile_id,
            person_profile_id=person_profile_id,
            normalized_username=normalized_username,
        )
    elif mode == "production":
        decision = _decide_production_mode(
            testing_profile_id=testing_profile_id,
            person_profile_id=person_profile_id,
        )
    else:
        # dry_run (and anything else) should not block candidate selection.
        decision = OutboundPolicyDecision(True)

    return decision


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner


def _test_testing_allowlist_by_display_name() -> bool:
    original_mode = config_schema.app_mode
    original_test_profile = config_schema.testing_profile_id
    original_test_username = config_schema.testing_username

    try:
        config_schema.app_mode = "testing"
        config_schema.testing_profile_id = None
        config_schema.testing_username = None

        allowed = SimpleNamespace(username="Frances McHardy", profile_id="X")
        blocked = SimpleNamespace(username="Some Other Person", profile_id="Y")

        assert should_allow_outbound_to_person(allowed).allowed is True
        decision = should_allow_outbound_to_person(blocked)
        assert decision.allowed is False
        assert decision.reason.startswith("skipped (testing_mode_filter:")
        return True
    finally:
        config_schema.app_mode = original_mode
        config_schema.testing_profile_id = original_test_profile
        config_schema.testing_username = original_test_username


def _test_testing_allowlist_by_profile_id_override() -> bool:
    original_mode = config_schema.app_mode
    original_test_profile = config_schema.testing_profile_id

    try:
        config_schema.app_mode = "testing"
        config_schema.testing_profile_id = "PROFILE_123"

        allowed = SimpleNamespace(username="Not Frances", profile_id="PROFILE_123")
        blocked = SimpleNamespace(username="Not Frances", profile_id="OTHER")

        assert should_allow_outbound_to_person(allowed).allowed is True
        assert should_allow_outbound_to_person(blocked).allowed is False
        return True
    finally:
        config_schema.app_mode = original_mode
        config_schema.testing_profile_id = original_test_profile


def _test_production_blocks_test_profile() -> bool:
    original_mode = config_schema.app_mode
    original_test_profile = config_schema.testing_profile_id

    try:
        config_schema.app_mode = "production"
        config_schema.testing_profile_id = "PROFILE_123"

        blocked = SimpleNamespace(username="Frances McHardy", profile_id="PROFILE_123")
        allowed = SimpleNamespace(username="Anyone", profile_id="OTHER")

        decision_blocked = should_allow_outbound_to_person(blocked)
        assert decision_blocked.allowed is False
        assert decision_blocked.reason.startswith("skipped (production_mode_filter:")
        assert should_allow_outbound_to_person(allowed).allowed is True
        return True
    finally:
        config_schema.app_mode = original_mode
        config_schema.testing_profile_id = original_test_profile


def module_tests() -> bool:
    suite = TestSuite("App Mode Policy", "core/app_mode_policy.py")
    suite.start_suite()
    suite.run_test(
        "Testing allowlist by display name",
        _test_testing_allowlist_by_display_name,
        "Allow only Frances McHardy/Milne (normalized) when no test overrides are set",
    )
    suite.run_test(
        "Testing allowlist by profile override",
        _test_testing_allowlist_by_profile_id_override,
        "Allow only TEST_PROFILE_ID in testing mode when configured",
    )
    suite.run_test(
        "Production blocks test profile",
        _test_production_blocks_test_profile,
        "Block sending to TEST_PROFILE_ID in production mode when configured",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
