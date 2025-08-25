#!/usr/bin/env python3
"""
Debug script to test the prefetch query logic for Action 8
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func

from core.session_manager import SessionManager
from database import ConversationLog, MessageDirectionEnum


def _get_latest_out_log_for_person(session, person_id: int):
    """Get the latest OUT log for a specific person."""
    latest_ts_subq = (
        session.query(
            ConversationLog.people_id,
            ConversationLog.direction,
            func.max(ConversationLog.latest_timestamp).label("max_ts"),
        )
        .filter(
            ConversationLog.people_id == person_id,
            ConversationLog.direction == MessageDirectionEnum.OUT,
            ~ConversationLog.conversation_id.like('template_tracking_%')
        )
        .group_by(ConversationLog.people_id, ConversationLog.direction)
        .subquery("latest_ts_subq")
    )

    return (
        session.query(ConversationLog)
        .join(
            latest_ts_subq,
            and_(
                ConversationLog.people_id == latest_ts_subq.c.people_id,
                ConversationLog.direction == latest_ts_subq.c.direction,
                ConversationLog.latest_timestamp == latest_ts_subq.c.max_ts,
            ),
        )
        .filter(~ConversationLog.conversation_id.like('template_tracking_%'))
        .first()
    )


def _analyze_prefetch_result(latest_out_log, person_id: int) -> None:
    """Analyze and display prefetch results for a person."""
    if latest_out_log:
        print('  ✓ Prefetch found latest OUT log:')
        print(f'    Timestamp: {latest_out_log.latest_timestamp}')
        print(f'    Conv ID: {latest_out_log.conversation_id}')
        print(f'    Message Type ID: {latest_out_log.message_type_id}')

        # Check if this should prevent messaging
        now_utc = datetime.now(timezone.utc)
        out_timestamp = latest_out_log.latest_timestamp

        # Apply timezone fix
        if out_timestamp.tzinfo is None:
            out_timestamp = out_timestamp.replace(tzinfo=timezone.utc)

        time_since_last = now_utc - out_timestamp
        MIN_MESSAGE_INTERVAL = timedelta(weeks=8)

        print(f'    Time since last: {time_since_last}')
        print(f'    MIN_MESSAGE_INTERVAL: {MIN_MESSAGE_INTERVAL}')

        if time_since_last < MIN_MESSAGE_INTERVAL:
            print('    ✅ SHOULD SKIP: Interval not met')
        else:
            print('    ❌ WOULD SEND: Interval met (this is wrong!)')
    else:
        print('  ❌ Prefetch found NO latest OUT log')
        print('    This means the person would get a message (first time)')


def debug_prefetch() -> None:
    """Debug the prefetch query logic for Action 8."""
    session_manager = SessionManager()
    session_manager.ensure_db_ready()

    with session_manager.get_db_conn_context() as session:
        if session:
            print('=== DEBUGGING PREFETCH QUERY LOGIC ===')
            print()

            # Test the exact prefetch query from Action 8 lines 1362-1376
            people_ids = [1, 3, 4, 5, 6]  # People who got duplicates

            for person_id in people_ids:
                print(f'--- Testing Person {person_id} ---')
                latest_out_log = _get_latest_out_log_for_person(session, person_id)
                _analyze_prefetch_result(latest_out_log, person_id)
                print()

            print('=== CONCLUSION ===')
            print('If prefetch is working correctly, all people should show "SHOULD SKIP"')
            print('If any show "WOULD SEND" or "NO latest OUT log", the prefetch is broken')

if __name__ == "__main__":
    debug_prefetch()
