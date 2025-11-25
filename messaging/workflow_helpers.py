"""Shared workflow helpers for messaging-centric actions.

This module centralizes the previously duplicated ``safe_column_value`` helpers
that lived in ``action7_inbox.py``, ``action8_messaging.py``, and
``action9_process_productive.py``. The helpers perform conservative type
coercion (bool/int/float/str/datetime) while optionally converting known enum
fields back into their Enum members. Keeping this logic in one place makes the
inbox/messaging stack easier to evolve and removes ~70 lines of duplicated
utilities across the three actions.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Protocol

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from config import config_schema
from test_framework import TestSuite, create_standard_test_runner

EnumOverrideMap = Mapping[str, type[Enum]]

_LOGGER = logging.getLogger(__name__)

# Optional import with proper typing - database may not be available during import
# Use Any for the enum type since we access .OUT/.IN dynamically
MessageDirectionEnum: Any = None
try:
    from database import MessageDirectionEnum as _MDE

    MessageDirectionEnum = _MDE
except Exception:  # pragma: no cover - optional dependency
    pass


class SafeColumnValueFunc(Protocol):
    def __call__(self, obj: Any, attr_name: str, default: Any | None = None) -> Any:  # pragma: no cover - protocol
        ...


def _coerce_enum(value: Any, enum_cls: type[Enum], default: Any) -> Any:
    """Convert ``value`` to ``enum_cls`` when possible."""

    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            return default
    return default


def _coerce_numeric_like(value: str) -> Optional[int | float]:
    """Attempt to coerce a numeric-looking string to int/float."""

    stripped = value.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        return None


def _coerce_value(value: Any, default: Any) -> Any:
    """Best-effort conversion of SQLAlchemy column-like values."""

    try:
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            numeric_candidate = _coerce_numeric_like(value)
            return numeric_candidate if numeric_candidate is not None else value
        if hasattr(value, "isoformat"):
            # datetime/date objects expose isoformat()
            return value
        return value
    except Exception:
        return default


def safe_column_value(
    obj: Any,
    attr_name: str,
    default: Any = None,
    *,
    enum_overrides: Optional[EnumOverrideMap] = None,
) -> Any:
    """Safely extract the attribute ``attr_name`` from ``obj``.

    ``enum_overrides`` maps attribute names to Enum classes for callers that need
    automatic conversion (e.g., ``{"status": PersonStatusEnum}``).
    """

    if obj is None or not hasattr(obj, attr_name):
        return default

    try:
        value = getattr(obj, attr_name)
    except Exception:
        return default

    if value is None:
        return default

    if enum_overrides and attr_name in enum_overrides:
        return _coerce_enum(value, enum_overrides[attr_name], default)

    return _coerce_value(value, default)


def build_safe_column_value(enum_overrides: Optional[EnumOverrideMap] = None) -> SafeColumnValueFunc:
    """Return a partially-applied ``safe_column_value`` for module-level reuse."""

    def _wrapped(obj: Any, attr_name: str, default: Any = None) -> Any:
        return safe_column_value(obj, attr_name, default=default, enum_overrides=enum_overrides)

    return _wrapped


def _get_conversation_state(person: Any) -> Any:
    return getattr(person, "conversation_state", None)


def log_conversation_state_change(
    person: Any,
    change_type: str,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    *,
    logger: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> None:
    """Log structured conversation-state transitions for observability."""

    conv_state = _get_conversation_state(person)
    if conv_state is None:
        return

    logger = logger or _LOGGER
    engagement = getattr(conv_state, "engagement_score", 0) or 0
    username = getattr(person, "username", "Unknown")
    person_id = getattr(person, "id", "?")

    if old_value and new_value:
        message = (
            f"\N{ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS} {log_prefix}: Conversation state change for "
            f"{username} (ID {person_id}): {change_type} '{old_value}' \u2192 '{new_value}' (engagement: {engagement})"
        )
    elif new_value:
        message = (
            f"\N{ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS} {log_prefix}: Conversation state change for "
            f"{username} (ID {person_id}): {change_type} \u2192 '{new_value}' (engagement: {engagement})"
        )
    else:
        message = (
            f"\N{ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS} {log_prefix}: Conversation state change for "
            f"{username} (ID {person_id}): {change_type} (engagement: {engagement})"
        )

    logger.info(message)


def cancel_pending_messages_on_status_change(
    person: Any,
    *,
    logger: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> bool:
    """Cancel out-of-tree follow-ups when a match is now in-tree."""

    logger = logger or _LOGGER
    conv_state = _get_conversation_state(person)
    if conv_state is None:
        logger.debug(f"{log_prefix}: No conversation_state to update")
        return False

    try:
        old_action = getattr(conv_state, "next_action", None)
        conv_state.next_action = "status_changed"
        conv_state.next_action_date = None

        log_conversation_state_change(
            person,
            "cancellation",
            old_action,
            "status_changed",
            logger=logger,
            log_prefix=log_prefix,
        )

        username = getattr(person, "username", "Unknown")
        person_id = getattr(person, "id", "?")
        logger.info(f"✅ Cancelled pending messages for {username} (ID {person_id}): Status changed to in-tree")
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        username = getattr(person, "username", "Unknown")
        person_id = getattr(person, "id", "?")
        logger.error(f"Error cancelling messages for {username} (ID {person_id}): {exc}")
        return False


def cancel_pending_on_reply(
    person: Any,
    *,
    logger: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> bool:
    """Switch conversation state to await reply when the recipient engages."""

    logger = logger or _LOGGER
    conv_state = _get_conversation_state(person)
    if conv_state is None:
        logger.debug(f"{log_prefix}: No conversation_state to update")
        return False

    try:
        old_phase = getattr(conv_state, "conversation_phase", None)
        old_action = getattr(conv_state, "next_action", None)

        conv_state.next_action = "await_reply"
        conv_state.next_action_date = None
        conv_state.conversation_phase = "active_dialogue"

        if old_phase != "active_dialogue":
            log_conversation_state_change(
                person,
                "phase",
                old_phase,
                "active_dialogue",
                logger=logger,
                log_prefix=log_prefix,
            )
        log_conversation_state_change(
            person,
            "next_action",
            old_action,
            "await_reply",
            logger=logger,
            log_prefix=log_prefix,
        )

        username = getattr(person, "username", "Unknown")
        person_id = getattr(person, "id", "?")
        logger.info(
            f"✅ Cancelled pending follow-ups for {username} (ID {person_id}): Switched to active dialogue mode"
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        username = getattr(person, "username", "Unknown")
        person_id = getattr(person, "id", "?")
        logger.error(f"Error cancelling follow-ups for {username} (ID {person_id}): {exc}")
        return False


def calculate_days_since_login(
    last_logged_in: Optional[datetime],
    log_prefix: str = "",
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """Return whole days since ``last_logged_in`` in UTC, handling naive tzinfo."""

    if not last_logged_in:
        return None

    logger = logger or _LOGGER
    try:
        now_utc = datetime.now(timezone.utc)
        if last_logged_in.tzinfo is None:
            last_logged_in = last_logged_in.replace(tzinfo=timezone.utc)
        elif last_logged_in.tzinfo != timezone.utc:
            last_logged_in = last_logged_in.astimezone(timezone.utc)
        return (now_utc - last_logged_in).days
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"{log_prefix}: Error calculating days since login: {exc}")
        return None


def determine_engagement_tier(
    engagement_score: int,
    days_since_login: Optional[int],
    thresholds: Mapping[str, int],
    intervals: Mapping[str, int],
) -> tuple[timedelta, str]:
    """Map engagement/login recency into tier + follow-up interval."""

    high_threshold = thresholds["high"]
    medium_threshold = thresholds["medium"]
    low_threshold = thresholds["low"]
    active_login_days = thresholds["active_login"]
    moderate_login_days = thresholds["moderate_login"]

    if engagement_score >= high_threshold and days_since_login is not None and days_since_login < active_login_days:
        return timedelta(days=intervals["high"]), "high"
    if engagement_score >= medium_threshold or (
        days_since_login is not None and days_since_login < moderate_login_days
    ):
        return timedelta(days=intervals["medium"]), "medium"
    if engagement_score >= low_threshold or (days_since_login is not None and days_since_login < 90):
        return timedelta(days=intervals["low"]), "low"
    return timedelta(days=intervals["none"]), "none"


def calculate_adaptive_interval(
    engagement_score: int,
    last_logged_in: Optional[datetime],
    log_prefix: str = "",
    *,
    logger: Optional[logging.Logger] = None,
) -> timedelta:
    """Production-only adaptive timing offset driven by engagement + login recency."""

    logger = logger or _LOGGER
    app_mode = getattr(config_schema, "app_mode", "production")
    if app_mode in {"testing", "dry_run"}:
        logger.debug(f"{log_prefix}: Adaptive timing disabled in {app_mode} mode (fixed interval only)")
        return timedelta(0)

    thresholds = {
        "high": getattr(config_schema, "engagement_high_threshold", 70),
        "medium": getattr(config_schema, "engagement_medium_threshold", 40),
        "low": getattr(config_schema, "engagement_low_threshold", 20),
        "active_login": getattr(config_schema, "login_active_threshold", 7),
        "moderate_login": getattr(config_schema, "login_moderate_threshold", 30),
    }

    intervals = {
        "high": getattr(config_schema, "followup_high_engagement_days", 7),
        "medium": getattr(config_schema, "followup_medium_engagement_days", 14),
        "low": getattr(config_schema, "followup_low_engagement_days", 21),
        "none": getattr(config_schema, "followup_no_engagement_days", 30),
    }

    days_since_login = calculate_days_since_login(last_logged_in, log_prefix, logger=logger)
    interval, tier = determine_engagement_tier(engagement_score, days_since_login, thresholds, intervals)
    logger.debug(
        f"{log_prefix}: Adaptive interval: {interval.days} days (tier={tier}, engagement={engagement_score}, "
        f"days_since_login={days_since_login})"
    )
    return interval


def is_tree_creation_recent(created_at: datetime, person: Any, *, logger: Optional[logging.Logger] = None) -> bool:
    """Check if a FamilyTree record was created within the configured recency window."""

    logger = logger or _LOGGER
    now_utc = datetime.now(timezone.utc)

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    elif created_at.tzinfo != timezone.utc:
        created_at = created_at.astimezone(timezone.utc)

    days_since_creation = (now_utc - created_at).days
    recent_threshold_days = getattr(config_schema, "status_change_recent_days", 7)

    if days_since_creation > recent_threshold_days:
        logger.debug(
            f"Person {getattr(person, 'username', 'Unknown')} (ID {getattr(person, 'id', '?')}): FamilyTree created "
            f"{days_since_creation} days ago (threshold: {recent_threshold_days} days) - not recent"
        )
        return False

    return True


def has_message_after_tree_creation(person: Any, created_at: datetime) -> bool:
    """Return True if an outgoing message was sent after ``created_at``."""

    entries = getattr(person, "conversation_log_entries", None)
    if not entries:
        return False

    for log in entries:
        direction = getattr(log, "direction", None)
        if MessageDirectionEnum is not None:
            is_outgoing = direction == MessageDirectionEnum.OUT
        else:  # Fallback to raw value when Enum unavailable
            is_outgoing = isinstance(direction, str) and direction.upper() == "OUT"
        if not is_outgoing:
            continue

        log_timestamp = getattr(log, "latest_timestamp", None)
        if log_timestamp is None:
            continue
        if log_timestamp.tzinfo is None:
            log_timestamp = log_timestamp.replace(tzinfo=timezone.utc)
        elif log_timestamp.tzinfo != timezone.utc:
            log_timestamp = log_timestamp.astimezone(timezone.utc)
        if log_timestamp > created_at:
            return True

    return False


def detect_status_change_to_in_tree(
    person: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Detect recent out-of-tree to in-tree transitions requiring message cancellations."""

    logger = logger or _LOGGER

    in_tree = getattr(person, "in_my_tree", False)
    family_tree = getattr(person, "family_tree", None)
    if not in_tree or not family_tree:
        return False

    try:
        created_at = getattr(family_tree, "created_at")
    except AttributeError:
        return False

    if not isinstance(created_at, datetime):
        return False

    if not is_tree_creation_recent(created_at, person, logger=logger):
        return False
    if has_message_after_tree_creation(person, created_at):
        return False

    try:
        created_days = (datetime.now(timezone.utc) - created_at).days
    except Exception:  # pragma: no cover - defensive
        created_days = "?"

    logger.info(
        f"✨ Status change detected: {getattr(person, 'username', 'Unknown')} (ID {getattr(person, 'id', '?')}) "
        f"recently added to tree ({created_days} days ago)"
    )
    return True


def calculate_follow_up_action(
    person: Any,
    conv_state: Any,
    log_prefix: str = "",
    *,
    logger: Optional[logging.Logger] = None,
) -> tuple[str, datetime | None]:
    """Return follow-up action tuple using adaptive timing helper."""

    logger = logger or _LOGGER
    engagement_score = getattr(conv_state, "engagement_score", 0) or 0
    interval = calculate_adaptive_interval(
        engagement_score,
        getattr(person, "last_logged_in", None),
        log_prefix,
        logger=logger,
    )

    if interval.total_seconds() == 0:
        logger.debug(f"{log_prefix}: No adaptive interval - no follow-up scheduled")
        return ("no_action", None)

    next_date = datetime.now() + interval
    logger.info(f"{log_prefix}: Follow-up scheduled in {interval.days} days (engagement: {engagement_score})")
    return ("send_follow_up", next_date)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _test_coerces_enum_members() -> bool:
    class SampleEnum(Enum):
        FIRST = "FIRST"
        SECOND = "SECOND"

    obj = SimpleNamespace(status="FIRST")
    func = build_safe_column_value({"status": SampleEnum})
    assert func(obj, "status") is SampleEnum.FIRST
    assert func(obj, "missing", default="none") == "none"
    obj.status = "UNKNOWN"
    assert func(obj, "status", default=SampleEnum.SECOND) is SampleEnum.SECOND
    return True


def _test_coerces_numeric_strings_and_primitives() -> bool:
    obj = SimpleNamespace(count="42", percent="3.14", flag=True, text=" hello ")
    base_safe = build_safe_column_value()
    assert base_safe(obj, "count") == 42
    assert abs(base_safe(obj, "percent") - 3.14) < 1e-6
    assert base_safe(obj, "flag") is True
    assert base_safe(obj, "text") == " hello "
    return True


def _test_handles_missing_attributes() -> bool:
    obj = SimpleNamespace()
    func = build_safe_column_value()
    assert func(obj, "unknown", default="n/a") == "n/a"
    obj.some = None
    assert func(obj, "some", default=0) == 0
    return True


def _test_safe_column_value_direct_call() -> bool:
    obj = SimpleNamespace(status="FIRST")

    class AnotherEnum(Enum):
        FIRST = "FIRST"
        SECOND = "SECOND"

    result = safe_column_value(obj, "status", enum_overrides={"status": AnotherEnum})
    assert result is AnotherEnum.FIRST
    return True


def _make_stub_person() -> SimpleNamespace:
    conv_state = SimpleNamespace(
        next_action="send_follow_up",
        next_action_date=datetime.now(timezone.utc) + timedelta(days=10),
        conversation_phase="initial_outreach",
        engagement_score=55,
    )
    return SimpleNamespace(username="Tester", id=1, conversation_state=conv_state)


def _test_cancel_pending_messages_helper() -> bool:
    person = _make_stub_person()
    assert cancel_pending_messages_on_status_change(person, log_prefix="test") is True
    assert person.conversation_state.next_action == "status_changed"
    assert person.conversation_state.next_action_date is None
    return True


def _test_cancel_pending_on_reply_helper() -> bool:
    person = _make_stub_person()
    person.conversation_state.conversation_phase = "initial_outreach"
    person.conversation_state.next_action = "send_follow_up"
    assert cancel_pending_on_reply(person, log_prefix="test") is True
    assert person.conversation_state.next_action == "await_reply"
    assert person.conversation_state.conversation_phase == "active_dialogue"
    return True


def _test_log_conversation_state_change_without_state() -> bool:
    person = SimpleNamespace(username="Tester", id=2, conversation_state=None)
    # Should not raise even without state or logger
    log_conversation_state_change(person, "phase")
    return True


def _test_calculate_days_since_login_handles_timezones() -> bool:
    base_now = datetime.now(timezone.utc)
    naive_login = (base_now - timedelta(days=5)).replace(tzinfo=None)
    aware_login = (base_now - timedelta(days=3)).astimezone(timezone(timedelta(hours=-5)))

    assert calculate_days_since_login(naive_login, "test") == 5
    assert calculate_days_since_login(aware_login, "test") == 3
    assert calculate_days_since_login(None, "test") is None
    return True


def _test_determine_engagement_tier_thresholds() -> bool:
    thresholds = {
        "high": 70,
        "medium": 40,
        "low": 20,
        "active_login": 7,
        "moderate_login": 30,
    }
    intervals = {"high": 7, "medium": 14, "low": 21, "none": 30}

    high_interval, high_tier = determine_engagement_tier(80, 3, thresholds, intervals)
    med_interval, med_tier = determine_engagement_tier(50, None, thresholds, intervals)
    low_interval, low_tier = determine_engagement_tier(10, 60, thresholds, intervals)

    assert (high_interval.days, high_tier) == (7, "high")
    assert (med_interval.days, med_tier) == (14, "medium")
    assert (low_interval.days, low_tier) == (21, "low")
    return True


def _test_is_tree_creation_recent_respects_threshold() -> bool:
    original_threshold = getattr(config_schema, "status_change_recent_days", 7)
    config_schema.status_change_recent_days = 10
    try:
        person = SimpleNamespace(username="Tester", id=1)
        recent_creation = datetime.now(timezone.utc) - timedelta(days=5)
        old_creation = datetime.now(timezone.utc) - timedelta(days=25)

        assert is_tree_creation_recent(recent_creation, person)
        assert is_tree_creation_recent(old_creation, person) is False
    finally:
        config_schema.status_change_recent_days = original_threshold
    return True


def _test_has_message_after_tree_creation_detects_activity() -> bool:
    created_at = datetime.now(timezone.utc) - timedelta(days=30)
    # Use actual enum value since MessageDirectionEnum is imported
    direction_out = MessageDirectionEnum.OUT if MessageDirectionEnum else "OUT"
    log_before = SimpleNamespace(direction=direction_out, latest_timestamp=created_at - timedelta(days=1))
    log_after = SimpleNamespace(direction=direction_out, latest_timestamp=created_at + timedelta(days=2))
    person = SimpleNamespace(username="Tester", id=1, conversation_log_entries=[log_before, log_after])
    assert has_message_after_tree_creation(person, created_at) is True

    person_without = SimpleNamespace(username="Tester", id=1, conversation_log_entries=[log_before])
    assert has_message_after_tree_creation(person_without, created_at) is False
    return True


def _test_calculate_follow_up_action() -> bool:
    original_mode = getattr(config_schema, "app_mode", "production")
    try:
        config_schema.app_mode = "production"
        person = SimpleNamespace(
            username="Follow-up Test User",
            id=303,
            last_logged_in=datetime.now() - timedelta(days=3),
        )
        conv_state = SimpleNamespace(engagement_score=75)

        action, next_date = calculate_follow_up_action(person, conv_state, "test")
        assert action == "send_follow_up"
        assert isinstance(next_date, datetime)
    finally:
        config_schema.app_mode = original_mode
    return True


def _test_cancel_pending_messages_updates_state() -> bool:
    conv_state = SimpleNamespace(
        next_action="send_follow_up",
        next_action_date=datetime.now(timezone.utc) + timedelta(days=10),
        engagement_score=55,
    )
    person = SimpleNamespace(username="Tester", id=1, conversation_state=conv_state)

    # Actually call the function and verify state changes
    result = cancel_pending_messages_on_status_change(person, log_prefix="test")

    assert result is True
    assert conv_state.next_action == "status_changed"
    assert conv_state.next_action_date is None
    return True


def module_tests() -> bool:
    suite = TestSuite("messaging.workflow_helpers", "messaging/workflow_helpers.py")
    suite.run_test("Enum coercion", _test_coerces_enum_members, "Ensures Enums convert via overrides.")
    suite.run_test("Numeric coercion", _test_coerces_numeric_strings_and_primitives, "Casts numeric strings.")
    suite.run_test("Missing attribute handling", _test_handles_missing_attributes, "Defaults for missing attrs.")
    suite.run_test("Direct helper usage", _test_safe_column_value_direct_call, "Direct helper supports overrides.")
    suite.run_test(
        "Cancel pending status helper",
        _test_cancel_pending_messages_helper,
        "Ensures conversation state is updated for status changes.",
    )
    suite.run_test(
        "Cancel on reply helper",
        _test_cancel_pending_on_reply_helper,
        "Switches to await_reply and active_dialogue modes.",
    )
    suite.run_test(
        "Log helper handles missing state",
        _test_log_conversation_state_change_without_state,
        "No-op when conversation state absent.",
    )
    suite.run_test(
        "Days since login helper",
        _test_calculate_days_since_login_handles_timezones,
        "Handles naive, aware, and missing timestamps.",
    )
    suite.run_test(
        "Engagement tier helper",
        _test_determine_engagement_tier_thresholds,
        "Validates tier selection and interval mapping.",
    )
    suite.run_test(
        "Tree creation recency helper",
        _test_is_tree_creation_recent_respects_threshold,
        "Checks configurable recency thresholds.",
    )
    suite.run_test(
        "Message-after-tree detection",
        _test_has_message_after_tree_creation_detects_activity,
        "Detects outgoing messages after tree creation.",
    )
    suite.run_test(
        "Follow-up action helper",
        _test_calculate_follow_up_action,
        "Calculates adaptive follow-up scheduling.",
    )
    suite.run_test(
        "Status change cancellation updates state",
        _test_cancel_pending_messages_updates_state,
        "Ensures cancellation helper updates next action and logs.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
