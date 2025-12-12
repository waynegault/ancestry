"""Action 11: Send approved drafts.

Sends only DraftReply rows that have been human-approved (status=APPROVED).
Optionally includes AUTO_APPROVED drafts when auto-approval is enabled.

This action:
- Re-checks outbound guardrails (conversation state hard stops, person automation flag, app-mode policy)
- Sends via the canonical API helper (call_send_message_api)
- Marks drafts SENT only on successful send
- Writes an OUT ConversationLog entry for auditability
- Updates ConversationMetrics and ConversationState without schema migrations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from api.api_utils import SEND_SUCCESS_DELIVERED, SEND_SUCCESS_DRY_RUN, call_send_message_api
from config import config_schema
from core.app_mode_policy import should_allow_outbound_to_person
from core.database import (
    ConversationLog,
    ConversationState,
    DraftReply,
    EngagementTracking,
    MessageDirectionEnum,
    Person,
    PersonStatusEnum,
)
from core.logging_utils import log_final_summary
from observability.conversation_analytics import record_engagement_event, update_conversation_metrics
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner, create_test_database

logger = logging.getLogger(__name__)


_SEND_SUCCESS_STATUSES: frozenset[str] = frozenset({SEND_SUCCESS_DELIVERED, SEND_SUCCESS_DRY_RUN})


@dataclass(slots=True)
class SendApprovedDraftsSummary:
    attempted: int = 0
    sent: int = 0
    skipped: int = 0
    errors: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)


def _normalize_enum(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value) or "").upper()


def _conversation_state_blocks_outbound(person: Person) -> Optional[str]:
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


def _person_blocks_outbound(person: Person) -> Optional[str]:
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


def _touch_conversation_state_after_send(db_session: Session, people_id: int) -> Optional[str]:
    """Lightweight state update: after an outbound send we are awaiting a reply."""

    conv_state = db_session.query(ConversationState).filter_by(people_id=people_id).first()
    if conv_state is None:
        return None

    conversation_phase = conv_state.conversation_phase

    # Avoid mutating hard-stop states.
    status_str = _normalize_enum(getattr(conv_state, "status", None))
    if status_str in {"OPT_OUT", "HUMAN_REVIEW", "PAUSED"}:
        return conversation_phase

    conv_state.next_action = "await_reply"
    conv_state.next_action_date = None
    db_session.add(conv_state)
    return conversation_phase


def _fetch_approved_drafts(db_session: Session, *, statuses: list[str], max_to_send: Optional[int]) -> list[DraftReply]:
    query = db_session.query(DraftReply).filter(DraftReply.status.in_(statuses)).order_by(DraftReply.created_at.asc())
    if max_to_send is not None and max_to_send > 0:
        query = query.limit(max_to_send)
    return query.all()


def _get_person_for_draft(db_session: Session, draft: DraftReply) -> tuple[Optional[Person], Optional[str]]:
    person = db_session.query(Person).filter(Person.id == draft.people_id).first()
    if person is None:
        return None, "person_missing"
    return person, None


def _get_message_text_for_draft(draft: DraftReply) -> tuple[Optional[str], Optional[str]]:
    from core.draft_content import strip_review_only_content

    message_text = strip_review_only_content(draft.content or "").strip()
    if not message_text:
        return None, "skipped (empty_draft)"
    return message_text, None


def _app_mode_blocks_outbound(person: Person, app_mode: str) -> Optional[str]:
    if app_mode == "dry_run":
        return None

    decision = should_allow_outbound_to_person(person, app_mode=app_mode)
    if decision.allowed:
        return None
    return decision.reason


def _send_single_approved_draft(
    *,
    db_session: Session,
    session_manager: Any,
    draft: DraftReply,
    app_mode: str,
    send_fn: Callable[[Any, Person, str, Optional[str], str], tuple[str, Optional[str]]],
) -> tuple[str, Optional[str]]:
    """Send one draft and persist results. Returns (outcome, reason)."""

    person, person_skip = _get_person_for_draft(db_session, draft)
    if person_skip:
        return "skipped", person_skip

    assert person is not None  # for type checkers

    block_reason = _person_blocks_outbound(person)
    if block_reason:
        return "skipped", f"skipped ({block_reason})"

    mode_skip = _app_mode_blocks_outbound(person, app_mode)
    if mode_skip:
        return "skipped", mode_skip

    message_text, text_skip = _get_message_text_for_draft(draft)
    if text_skip:
        return "skipped", text_skip

    assert message_text is not None

    existing_conv_id = (draft.conversation_id or "").strip() or None
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

    now = datetime.now(timezone.utc)
    effective_id = effective_conv_id or existing_conv_id or f"draft_{draft.id}"
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
    draft.status = "SENT"
    db_session.add(draft)
    conversation_phase = _touch_conversation_state_after_send(db_session, person.id)
    db_session.commit()

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

    update_conversation_metrics(db_session, people_id=person.id, message_sent=True)
    return "sent", ""


def run_send_approved_drafts(
    *,
    db_session: Session,
    session_manager: Any,
    max_to_send: Optional[int] = None,
    include_auto_approved: bool = False,
    send_fn: Callable[[Any, Person, str, Optional[str], str], tuple[str, Optional[str]]] = call_send_message_api,
) -> SendApprovedDraftsSummary:
    """Core runner implementation (callable from Action wrapper and tests)."""

    summary = SendApprovedDraftsSummary()

    if bool(getattr(config_schema, "emergency_stop_enabled", False)):
        logger.warning("Emergency stop enabled; refusing to send drafts")
        return summary

    app_mode = str(getattr(config_schema, "app_mode", "dry_run") or "dry_run").strip().lower()

    statuses: list[str] = ["APPROVED"]
    if include_auto_approved:
        statuses.append("AUTO_APPROVED")

    drafts = _fetch_approved_drafts(db_session, statuses=statuses, max_to_send=max_to_send)
    for draft in drafts:
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
            logger.error("DraftReply #%s: unexpected exception: %s", getattr(draft, "id", "?"), exc, exc_info=True)
            continue

        if outcome == "sent":
            summary.sent += 1
        elif outcome == "skipped":
            summary.skipped += 1
            reason_key = reason or "skipped (unknown)"
            summary.skip_reasons[reason_key] = summary.skip_reasons.get(reason_key, 0) + 1
        else:
            summary.errors += 1

    return summary


def send_approved_drafts(session_manager: Any, *_: Any) -> bool:
    """Action entrypoint (invoked via exec_actn)."""

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
            _sm: Any, _person: Person, _text: str, _conv: Optional[str], _lp: str
        ) -> tuple[str, Optional[str]]:
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

    def fake_send(*_: Any, **__: Any) -> tuple[str, Optional[str]]:
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

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    raise SystemExit(0 if run_comprehensive_tests() else 1)
