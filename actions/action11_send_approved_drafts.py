"""Action 11: Send approved drafts.

Sends only DraftReply rows that have been human-approved (status=APPROVED).
Optionally includes AUTO_APPROVED drafts when auto-approval is enabled.

This action:
- Re-checks outbound guardrails (conversation state hard stops, person automation flag, app-mode policy)
- Sends via the canonical API helper (call_send_message_api)
- Marks drafts SENT only on successful send
- Writes an OUT ConversationLog entry for auditability
- Updates ConversationMetrics and ConversationState without schema migrations
- Supports feature flags for gradual rollout (ACTION11_SEND_ENABLED)
"""


import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import SimpleNamespace
from typing import Any

from sqlalchemy.orm import Session

from api.api_utils import SEND_SUCCESS_DELIVERED, SEND_SUCCESS_DRY_RUN, call_send_message_api
from config import config_schema
from core.app_mode_policy import should_allow_outbound_to_person
from core.circuit_breaker import SessionCircuitBreaker
from core.database import (
    ConversationLog,
    ConversationState,
    DraftReply,
    EngagementTracking,
    MessageDirectionEnum,
    Person,
    PersonStatusEnum,
)
from core.feature_flags import is_feature_enabled
from core.logging_utils import log_final_summary
from messaging.send_orchestrator import (
    MessageSendOrchestrator,
    create_action11_context,
    should_use_orchestrator_for_action11,
)
from observability.conversation_analytics import record_engagement_event, update_conversation_metrics
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner, create_test_database

logger = logging.getLogger(__name__)


class SendErrorCategory(Enum):
    """Categorization of send failure types for monitoring and debugging.

    Categories help identify root causes:
    - NETWORK: Transient connectivity issues (retry recommended)
    - AUTH: Authentication failures (session refresh needed)
    - RATE_LIMIT: API throttling (backoff required)
    - VALIDATION: Pre-send checks failed (skip this draft)
    - API_ERROR: Server-side error from Ancestry API
    - TRANSACTION: Database commit failure
    - UNKNOWN: Uncategorized exception
    """

    NETWORK = "network"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    API_ERROR = "api_error"
    TRANSACTION = "transaction"
    UNKNOWN = "unknown"


def categorize_send_error(exc: Exception) -> SendErrorCategory:
    """Categorize an exception into a send error type.

    Args:
        exc: The exception to categorize

    Returns:
        SendErrorCategory for monitoring and retry decisions
    """
    error_msg = str(exc).lower()
    error_type = type(exc).__name__.lower()

    checks: list[tuple[Iterable[str], SendErrorCategory, str]] = [
        (["timeout", "connection", "network", "socket"], SendErrorCategory.NETWORK, error_msg),
        (["401", "403", "unauthorized", "forbidden", "auth"], SendErrorCategory.AUTH, error_msg),
        (["429", "rate", "throttle", "too many"], SendErrorCategory.RATE_LIMIT, error_msg),
        (["500", "502", "503", "504", "server error"], SendErrorCategory.API_ERROR, error_msg),
        (["sqlalchemy", "database", "transaction", "commit"], SendErrorCategory.TRANSACTION, error_type),
        (["validation", "invalid", "missing required"], SendErrorCategory.VALIDATION, error_msg),
    ]

    for terms, category, haystack in checks:
        if any(term in haystack for term in terms):
            return category

    return SendErrorCategory.UNKNOWN


_SEND_SUCCESS_STATUSES: frozenset[str] = frozenset({SEND_SUCCESS_DELIVERED, SEND_SUCCESS_DRY_RUN})


@dataclass(slots=True)
class SendApprovedDraftsSummary:
    attempted: int = 0
    sent: int = 0
    skipped: int = 0
    errors: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    error_categories: dict[str, int] = field(default_factory=dict)


def _normalize_enum(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value) or "").upper()


def _conversation_state_blocks_outbound(person: Person) -> str | None:
    conv_state = getattr(person, "conversation_state", None)
    if not conv_state:
        return None

    status_str = _normalize_enum(getattr(conv_state, "status", None))
    state_str = _normalize_enum(getattr(conv_state, "state", None))
    active_state = next((s for s in (status_str, state_str) if s), "")

    blocked_states = {"OPT_OUT", "HUMAN_REVIEW", "PAUSED", "DESIST"}
    if active_state in blocked_states:
        return f"conversation_state_{active_state.lower()}"

    if bool(getattr(conv_state, "safety_flag", False)):
        return "conversation_state_safety"

    return None


def _person_blocks_outbound(person: Person) -> str | None:
    status_str = _normalize_enum(getattr(person, "status", None))
    if status_str in {"DESIST", "ARCHIVE", "BLOCKED"}:
        return f"person_status_{status_str.lower()}"

    if not bool(getattr(person, "contactable", True)):
        return "person_not_contactable"

    if not bool(getattr(person, "automation_enabled", True)):
        return "automation_disabled"

    conv_reason = _conversation_state_blocks_outbound(person)
    if conv_reason:
        return conv_reason

    return None


def _touch_conversation_state_after_send(db_session: Session, people_id: int) -> str | None:
    """
    Update conversation state after successful outbound send.

    Per the reply_management.md state machine:
    - After sending, we transition to 'awaiting_reply' phase
    - The status remains ACTIVE (unless in a hard-stop state)
    - next_action is set to 'await_reply'
    - updated_at is refreshed to track when the send occurred

    Phase 1.6.4 Implementation: ConversationState Synchronization
    """
    from datetime import datetime

    conv_state = db_session.query(ConversationState).filter_by(people_id=people_id).first()
    if conv_state is None:
        return None

    old_phase = conv_state.conversation_phase

    # Avoid mutating hard-stop states
    status_str = _normalize_enum(getattr(conv_state, "status", None))
    if status_str in {"OPT_OUT", "HUMAN_REVIEW", "PAUSED", "COMPLETED"}:
        logger.debug(
            "Skipping phase update for person %d: status=%s is a hard-stop state",
            people_id,
            status_str,
        )
        return old_phase

    # Update phase based on current state per state machine
    # initial_outreach -> awaiting_reply (first message sent)
    # active_dialogue -> awaiting_reply (follow-up sent)
    # Any other phase -> awaiting_reply (we just sent a message)
    new_phase = "awaiting_reply"

    conv_state.conversation_phase = new_phase
    conv_state.next_action = "await_reply"
    conv_state.next_action_date = None
    conv_state.updated_at = datetime.now(UTC)
    db_session.add(conv_state)

    logger.debug(
        "Updated conversation state for person %d: phase %s -> %s, next_action=%s",
        people_id,
        old_phase,
        new_phase,
        "await_reply",
    )

    return new_phase


def _fetch_approved_drafts(db_session: Session, *, statuses: list[str], max_to_send: int | None) -> list[DraftReply]:
    query = db_session.query(DraftReply).filter(DraftReply.status.in_(statuses)).order_by(DraftReply.created_at.asc())
    if max_to_send is not None and max_to_send > 0:
        query = query.limit(max_to_send)
    return query.all()


def _get_person_for_draft(db_session: Session, draft: DraftReply) -> tuple[Person | None, str | None]:
    person = db_session.query(Person).filter(Person.id == draft.people_id).first()
    if person is None:
        return None, "person_missing"
    return person, None


def _get_message_text_for_draft(draft: DraftReply) -> tuple[str | None, str | None]:
    from core.draft_content import strip_review_only_content

    message_text = strip_review_only_content(draft.content or "").strip()
    if not message_text:
        return None, "skipped (empty_draft)"
    return message_text, None


def _app_mode_blocks_outbound(person: Person, app_mode: str) -> str | None:
    if app_mode == "dry_run":
        return None

    decision = should_allow_outbound_to_person(person, app_mode=app_mode)
    if decision.allowed:
        return None
    return decision.reason


def _check_duplicate_send(db_session: Session, draft: DraftReply, person_id: int) -> str | None:
    """
    Phase 1.6.3: Duplicate Send Prevention

    Check if this draft was already sent or if there's a recent outbound message
    to this person (idempotency window).

    Returns:
        None if send should proceed, or a skip reason string if duplicate detected.
    """
    from datetime import timedelta

    # Guard 1: Check if draft is already SENT
    if draft.status == "SENT":
        logger.info(
            "Duplicate prevention: DraftReply #%d already has status SENT, skipping",
            draft.id,
        )
        return "already_sent"

    # Guard 2: Check for recent outbound log to same person (idempotency window: 5 minutes)
    idempotency_window = timedelta(minutes=5)
    cutoff = datetime.now(UTC) - idempotency_window

    recent_outbound = (
        db_session.query(ConversationLog)
        .filter(
            ConversationLog.people_id == person_id,
            ConversationLog.direction == MessageDirectionEnum.OUT,
            ConversationLog.latest_timestamp > cutoff,
        )
        .first()
    )

    if recent_outbound:
        logger.info(
            "Duplicate prevention: Recent outbound message found for person #%d within %s, skipping draft #%d",
            person_id,
            idempotency_window,
            draft.id,
        )
        return f"recent_outbound_within_{int(idempotency_window.total_seconds())}s"

    return None


def _gather_send_context(
    db_session: Session,
    draft: DraftReply,
    app_mode: str,
) -> tuple[Person | None, str | None, str | None, str | None]:
    """Collect pre-send context and detect skip reasons."""

    person, person_skip = _get_person_for_draft(db_session, draft)
    if person_skip:
        return None, None, None, person_skip

    assert person is not None

    duplicate_reason = _check_duplicate_send(db_session, draft, person.id)
    if duplicate_reason:
        return person, None, None, f"skipped ({duplicate_reason})"

    block_reason = _person_blocks_outbound(person)
    if block_reason:
        return person, None, None, f"skipped ({block_reason})"

    mode_skip = _app_mode_blocks_outbound(person, app_mode)
    if mode_skip:
        return person, None, None, mode_skip

    message_text, text_skip = _get_message_text_for_draft(draft)
    if text_skip:
        return person, None, None, text_skip

    existing_conv_id = (draft.conversation_id or "").strip() or None
    return person, message_text, existing_conv_id, None


def _run_shadow_mode_comparison(
    session_manager: Any,
    draft: DraftReply,
    person: Person,
    message_text: str,
) -> None:
    """
    Run shadow mode comparison between legacy and orchestrator decisions.

    This runs when orchestrator is NOT enabled but shadow mode IS enabled.
    """
    try:
        from messaging.send_orchestrator import create_action11_context
        from messaging.shadow_mode_analyzer import LegacyDecision, ShadowModeAnalyzer

        analyzer = ShadowModeAnalyzer(session_manager)
        if not analyzer.is_enabled:
            return

        # Create legacy decision - Action 11 sends if we get here
        legacy_decision = LegacyDecision(
            action_name="Action11",
            should_send=True,  # If we reach this point, legacy would send
            block_reason=None,
            person_id=person.id,
            trigger_type="HUMAN_APPROVED",
        )

        # Create context for orchestrator
        context = create_action11_context(
            person=person,
            conversation_logs=[],
            draft_content=message_text,
            draft_id=draft.id,
        )

        # Run shadow comparison
        analyzer.run_shadow_check(context, legacy_decision)

    except Exception as e:
        logger.debug(f"[SHADOW] Shadow mode comparison failed: {e}")


def _send_via_orchestrator(
    *,
    db_session: Session,
    session_manager: Any,
    draft: DraftReply,
    person: Person,
    message_text: str,
    _existing_conv_id: str | None,  # Reserved for future use
) -> tuple[str, str | None] | None:
    """
    Try to send via orchestrator if enabled.

    Returns:
        ("sent", None): Orchestrator sent successfully
        ("skipped", reason): Orchestrator blocked send (expected behavior)
        None: Orchestrator not enabled, caller should use legacy path
    """
    if not should_use_orchestrator_for_action11():
        return None

    try:
        # Get conversation logs for context
        conv_logs = (
            db_session.query(ConversationLog)
            .filter(ConversationLog.people_id == person.id)
            .order_by(ConversationLog.latest_timestamp.desc())
            .limit(10)
            .all()
        )

        # Create orchestrator context
        context = create_action11_context(
            person=person,
            conversation_logs=conv_logs,
            draft_content=message_text,
            draft_id=draft.id,
        )

        # Send via orchestrator
        orchestrator = MessageSendOrchestrator(session_manager)
        result = orchestrator.send(context)

        if result.success:
            log_prefix = f"DraftReply #{draft.id} Person #{person.id}"
            logger.info("%s: ✅ Orchestrator sent draft successfully", log_prefix)
            return ("sent", None)
        return ("skipped", result.error or "orchestrator_blocked")

    except Exception as e:
        logger.error("Orchestrator error for DraftReply #%s, falling back to legacy: %s", draft.id, e)
        return None  # Fall back to legacy path


def _send_single_approved_draft(
    *,
    db_session: Session,
    session_manager: Any,
    draft: DraftReply,
    app_mode: str,
    send_fn: Callable[[Any, Person, str, str | None, str], tuple[str, str | None]],
) -> tuple[str, str | None]:
    """Send one draft and persist results. Returns (outcome, reason)."""
    person, message_text, existing_conv_id, skip_reason = _gather_send_context(db_session, draft, app_mode)
    if skip_reason:
        return "skipped", skip_reason

    assert person is not None
    assert message_text is not None

    # Try orchestrator first (Phase 3 integration)
    orchestrator_result = _send_via_orchestrator(
        db_session=db_session,
        session_manager=session_manager,
        draft=draft,
        person=person,
        message_text=message_text,
        _existing_conv_id=existing_conv_id,
    )
    if orchestrator_result is not None:
        return orchestrator_result

    # Run shadow mode comparison if enabled (Phase 4.3)
    _run_shadow_mode_comparison(session_manager, draft, person, message_text)

    # Legacy path: direct API call
    log_prefix = f"DraftReply #{draft.id} Person #{person.id}"

    message_status, effective_conv_id = send_fn(
        session_manager,
        person,
        message_text,
        existing_conv_id,
        log_prefix,
    )

    if message_status not in _SEND_SUCCESS_STATUSES:
        logger.warning("%s: send failed (%s)", log_prefix, message_status)
        return "error", message_status

    # Phase 1.6.1: Transaction Safety
    # All database updates within a single transaction with proper rollback
    now = datetime.now(UTC)
    effective_id = effective_conv_id or existing_conv_id or f"draft_{draft.id}"

    try:
        # Create conversation log
        log = ConversationLog(
            conversation_id=effective_id,
            direction=MessageDirectionEnum.OUT,
            people_id=person.id,
            latest_message_content=message_text[: int(getattr(config_schema, "message_truncation_length", 1000))],
            latest_timestamp=now,
            ai_sentiment=None,
            message_template_id=None,
            script_message_status=f"{message_status} | Source: approved_draft (draft_id={draft.id})",
        )
        db_session.add(log)

        # Update draft status
        draft.status = "SENT"
        db_session.add(draft)

        # Update conversation state
        conversation_phase = _touch_conversation_state_after_send(db_session, person.id)

        # Commit the core send transaction
        db_session.commit()

    except Exception as exc:
        # Rollback on any failure - draft stays APPROVED for retry
        logger.error(
            "%s: Transaction failed, rolling back. Draft remains APPROVED. Error: %s",
            log_prefix,
            exc,
        )
        db_session.rollback()
        return "error", f"transaction_failed: {exc}"

    # Record engagement event (non-critical, separate transaction)
    try:
        record_engagement_event(
            session=db_session,
            people_id=person.id,
            event_type="message_sent",
            event_description="Approved draft sent",
            event_data={
                "source": "approved_draft",
                "draft_id": draft.id,
                "conversation_id": effective_id,
                "send_status": message_status,
                "app_mode": app_mode,
            },
            conversation_phase=conversation_phase,
        )
    except Exception as exc:
        # Non-critical: log failure but don't rollback the send
        logger.warning("%s: Failed to record engagement event: %s", log_prefix, exc)

    # Update metrics (non-critical, separate transaction)
    try:
        update_conversation_metrics(db_session, people_id=person.id, message_sent=True)
    except Exception as exc:
        # Non-critical: log failure but don't rollback the send
        logger.warning("%s: Failed to update conversation metrics: %s", log_prefix, exc)

    return "sent", ""


def run_send_approved_drafts(
    *,
    db_session: Session,
    session_manager: Any,
    max_to_send: int | None = None,
    include_auto_approved: bool = False,
    send_fn: Callable[[Any, Person, str, str | None, str], tuple[str, str | None]] = call_send_message_api,
) -> SendApprovedDraftsSummary:
    """Core runner implementation (callable from Action wrapper and tests).

    Uses circuit breaker pattern to fail fast after consecutive failures,
    preventing wasted API calls when the service is unavailable.
    """

    summary = SendApprovedDraftsSummary()

    if bool(getattr(config_schema, "emergency_stop_enabled", False)):
        logger.warning("Emergency stop enabled; refusing to send drafts")
        return summary

    app_mode = str(getattr(config_schema, "app_mode", "dry_run") or "dry_run").strip().lower()

    # Circuit breaker: fail fast after 5 consecutive send failures
    circuit_breaker = SessionCircuitBreaker(
        name="action11_send",
        threshold=5,
        recovery_timeout_sec=300,  # 5 minutes before retry
    )

    statuses: list[str] = ["APPROVED"]
    if include_auto_approved:
        statuses.append("AUTO_APPROVED")

    drafts = _fetch_approved_drafts(db_session, statuses=statuses, max_to_send=max_to_send)
    for draft in drafts:
        # Circuit breaker check: abort remaining drafts if tripped
        if circuit_breaker.is_tripped():
            logger.warning(
                "🚨 Circuit breaker TRIPPED after %d consecutive failures - aborting remaining %d drafts",
                circuit_breaker.threshold,
                len(drafts) - summary.attempted,
            )
            break

        summary.attempted += 1

        try:
            outcome, reason = _send_single_approved_draft(
                db_session=db_session,
                session_manager=session_manager,
                draft=draft,
                app_mode=app_mode,
                send_fn=send_fn,
            )
        except Exception as exc:  # pragma: no cover - defensive
            summary.errors += 1
            error_category = categorize_send_error(exc)
            summary.error_categories[error_category.value] = summary.error_categories.get(error_category.value, 0) + 1
            circuit_breaker.record_failure()
            logger.error(
                "DraftReply #%s: %s error (%s): %s",
                getattr(draft, "id", "?"),
                error_category.value,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            continue

        _handle_draft_outcome(outcome, reason, summary, circuit_breaker)

    return summary


def _handle_draft_outcome(
    outcome: str,
    reason: str | None,
    summary: SendApprovedDraftsSummary,
    circuit_breaker: SessionCircuitBreaker,
) -> None:
    if outcome == "sent":
        summary.sent += 1
        circuit_breaker.record_success()
        _record_drafts_sent_metric("sent")
        return

    if outcome == "skipped":
        summary.skipped += 1
        reason_key = reason or "skipped (unknown)"
        summary.skip_reasons[reason_key] = summary.skip_reasons.get(reason_key, 0) + 1
        _record_drafts_sent_metric("skipped")
        return

    summary.errors += 1
    circuit_breaker.record_failure()
    _record_drafts_sent_metric("error")


def _record_drafts_sent_metric(outcome: str) -> None:
    try:
        from observability.metrics_registry import metrics

        metrics().drafts_sent.inc(outcome=outcome)
    except Exception:
        pass  # Metrics are non-critical


def send_approved_drafts(session_manager: Any, *_: Any) -> bool:
    """Action entrypoint (invoked via exec_actn).

    Feature Flags:
        ACTION11_SEND_ENABLED: Master kill-switch for sending (default: True)
            Set FEATURE_FLAG_ACTION11_SEND_ENABLED=false to disable sending globally
    """

    # Feature flag check for gradual rollout / emergency disable
    if not is_feature_enabled("ACTION11_SEND_ENABLED", default=True):
        logger.warning("🚫 Action 11 disabled via feature flag ACTION11_SEND_ENABLED")
        return False

    max_send_per_run = int(getattr(config_schema, "max_send_per_run", 0) or 0)

    app_mode = str(getattr(config_schema, "app_mode", "dry_run") or "dry_run").strip().lower()
    allow_prod_auto = bool(getattr(config_schema, "allow_production_auto_approve", False))
    auto_enabled = bool(getattr(config_schema, "auto_approve_enabled", False))
    include_auto = auto_enabled and (app_mode != "production" or allow_prod_auto)

    with session_manager.get_db_conn_context() as db_session:
        if db_session is None:
            logger.error("Database session unavailable")
            return False

        start_time = time.time()
        summary = run_send_approved_drafts(
            db_session=db_session,
            session_manager=session_manager,
            max_to_send=max_send_per_run if max_send_per_run > 0 else None,
            include_auto_approved=include_auto,
        )
        duration_sec = time.time() - start_time

    log_final_summary(
        summary_dict={
            "App Mode": app_mode,
            "Include AUTO_APPROVED": include_auto,
            "Drafts Attempted": summary.attempted,
            "Drafts Sent": summary.sent,
            "Drafts Skipped": summary.skipped,
            "Errors": summary.errors,
            "Skip Reasons": (summary.skip_reasons or {}),
        },
        run_time_seconds=float(duration_sec),
    )

    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _test_sends_and_marks_sent() -> bool:
    original_mode = config_schema.app_mode
    original_max = config_schema.max_send_per_run
    original_auto = config_schema.auto_approve_enabled

    try:
        config_schema.app_mode = "dry_run"
        config_schema.max_send_per_run = 50
        config_schema.auto_approve_enabled = False

        session = create_test_database()
        person = Person(username="Test User", profile_id="PROFILE_1")
        session.add(person)
        session.commit()

        from core.draft_content import DraftInternalMetadata, append_internal_metadata

        legacy_with_appendix = "Hello there\n\n---\nResearch Suggestions:\nInternal-only suggestion"
        draft_content = append_internal_metadata(
            legacy_with_appendix,
            DraftInternalMetadata(
                ai_confidence=95,
                ai_reasoning="High confidence (test)",
                context_summary="Some internal context (test)",
                research_suggestions="Suggestion A\nSuggestion B",
                research_metadata={"shared_match_count": 3},
            ),
        )

        draft = DraftReply(
            people_id=person.id,
            conversation_id="conv_123",
            content=draft_content,
            status="APPROVED",
        )
        session.add(draft)
        session.commit()

        def fake_send(
            _sm: Any, _person: Person, _text: str, _conv: str | None, _lp: str
        ) -> tuple[str, str | None]:
            assert _text == "Hello there", "Review-only content must be stripped before send"
            return SEND_SUCCESS_DRY_RUN, "conv_123"

        summary = run_send_approved_drafts(
            db_session=session,
            session_manager=SimpleNamespace(),
            max_to_send=None,
            include_auto_approved=False,
            send_fn=fake_send,
        )

        updated = session.query(DraftReply).filter(DraftReply.id == draft.id).first()
        assert updated is not None
        assert updated.status == "SENT"

        out_logs = session.query(ConversationLog).filter(ConversationLog.people_id == person.id).all()
        assert len(out_logs) == 1
        assert out_logs[0].direction == MessageDirectionEnum.OUT
        assert out_logs[0].latest_message_content == "Hello there"

        from core.database import ConversationMetrics

        metrics = session.query(ConversationMetrics).filter_by(people_id=person.id).first()
        assert metrics is not None
        assert metrics.messages_sent == 1

        events = session.query(EngagementTracking).filter(EngagementTracking.people_id == person.id).all()
        assert any(e.event_type == "message_sent" for e in events), "Should record a message_sent engagement event"

        assert summary.sent == 1
        assert summary.errors == 0
        return True
    finally:
        config_schema.app_mode = original_mode
        config_schema.max_send_per_run = original_max
        config_schema.auto_approve_enabled = original_auto


def _test_skips_desist_people() -> bool:
    session = create_test_database()
    person = Person(username="No Contact", profile_id="PROFILE_2", status=PersonStatusEnum.DESIST)
    session.add(person)
    session.commit()

    draft = DraftReply(
        people_id=person.id,
        conversation_id="conv_456",
        content="Should never send",
        status="APPROVED",
    )
    session.add(draft)
    session.commit()

    def fake_send(*_: Any, **__: Any) -> tuple[str, str | None]:
        raise AssertionError("send_fn should not be called for DESIST")

    summary = run_send_approved_drafts(
        db_session=session,
        session_manager=SimpleNamespace(),
        max_to_send=None,
        include_auto_approved=False,
        send_fn=fake_send,
    )

    updated = session.query(DraftReply).filter(DraftReply.id == draft.id).first()
    assert updated is not None
    assert updated.status == "APPROVED"
    assert summary.sent == 0
    assert summary.skipped == 1
    return True


def _test_error_categorization() -> bool:
    """Test that send errors are correctly categorized."""
    # Network errors
    assert categorize_send_error(Exception("Connection timeout")) == SendErrorCategory.NETWORK
    assert categorize_send_error(Exception("socket error")) == SendErrorCategory.NETWORK

    # Auth errors
    assert categorize_send_error(Exception("401 Unauthorized")) == SendErrorCategory.AUTH
    assert categorize_send_error(Exception("403 Forbidden")) == SendErrorCategory.AUTH

    # Rate limiting
    assert categorize_send_error(Exception("429 Too Many Requests")) == SendErrorCategory.RATE_LIMIT
    assert categorize_send_error(Exception("Rate limit exceeded")) == SendErrorCategory.RATE_LIMIT

    # API errors
    assert categorize_send_error(Exception("500 Internal Server Error")) == SendErrorCategory.API_ERROR
    assert categorize_send_error(Exception("503 Service Unavailable")) == SendErrorCategory.API_ERROR

    # Unknown (default)
    assert categorize_send_error(Exception("Something weird")) == SendErrorCategory.UNKNOWN

    return True


def module_tests() -> bool:
    suite = TestSuite("Action 11 - Send Approved Drafts", "actions/action11_send_approved_drafts.py")
    suite.start_suite()

    suite.run_test(
        "Approved draft is sent + marked SENT",
        _test_sends_and_marks_sent,
        test_summary="Runner sends APPROVED draft, logs OUT message, updates metrics, and marks draft SENT.",
        functions_tested="run_send_approved_drafts",
        method_description="In-memory DB + stub send_fn returning dry-run success",
        expected_outcome="DraftReply.status becomes SENT and ConversationMetrics.messages_sent increments",
    )
    suite.run_test(
        "DESIST person is skipped",
        _test_skips_desist_people,
        test_summary="Runner must not send approved drafts to DESIST people.",
        functions_tested="run_send_approved_drafts",
        method_description="In-memory DB with DESIST Person and APPROVED draft; send_fn should not be called",
        expected_outcome="Draft remains APPROVED and summary records a skip",
    )
    suite.run_test(
        "Error categorization works correctly",
        _test_error_categorization,
        test_summary="Send errors are categorized by type (network, auth, rate_limit, etc.)",
        functions_tested="categorize_send_error, SendErrorCategory",
        method_description="Test various exception messages and verify correct category",
        expected_outcome="Each error type returns the expected SendErrorCategory",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    raise SystemExit(0 if run_comprehensive_tests() else 1)
