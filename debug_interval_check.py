#!/usr/bin/env python3
"""
Debug script to test the interval check logic for Action 8
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func

from core.session_manager import SessionManager
from database import ConversationLog, MessageDirectionEnum


def debug_interval_check() -> None:
    session_manager = SessionManager()
    session_manager.ensure_db_ready()

    with session_manager.get_db_conn_context() as session:
        if session:
            print('=== DEBUGGING INTERVAL CHECK LOGIC ===')
            print()

            # Test the exact logic from Action 8
            MIN_MESSAGE_INTERVAL = timedelta(weeks=8)
            now_utc = datetime.now(timezone.utc)

            print(f'Current time: {now_utc}')
            print(f'MIN_MESSAGE_INTERVAL: {MIN_MESSAGE_INTERVAL}')
            print()

            # Test Person 1 (Frances Mc Hardy) who got duplicates
            person_id = 1

            # Simulate the prefetch query for latest OUT log
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

            # Get the latest OUT log
            latest_out_log = (
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

            if latest_out_log:
                print(f'Found latest OUT log for Person {person_id}:')
                print(f'  Timestamp: {latest_out_log.latest_timestamp}')
                print(f'  Conv ID: {latest_out_log.conversation_id}')

                # Test the interval check logic
                out_timestamp = latest_out_log.latest_timestamp
                if out_timestamp:
                    # Fix timezone issue - make out_timestamp timezone-aware if it's naive
                    if out_timestamp.tzinfo is None:
                        out_timestamp = out_timestamp.replace(tzinfo=timezone.utc)
                    time_since_last = now_utc - out_timestamp
                    print(f'  Time since last: {time_since_last}')
                    print(f'  MIN_MESSAGE_INTERVAL: {MIN_MESSAGE_INTERVAL}')

                    if time_since_last < MIN_MESSAGE_INTERVAL:
                        print(f'  ✅ SHOULD SKIP: Interval not met ({time_since_last} < {MIN_MESSAGE_INTERVAL})')
                    else:
                        print(f'  ❌ WOULD SEND: Interval met ({time_since_last} >= {MIN_MESSAGE_INTERVAL})')
                else:
                    print('  ❌ ERROR: No timestamp found')
            else:
                print(f'❌ ERROR: No latest OUT log found for Person {person_id}')

            print()
            print('=== CONCLUSION ===')
            print('If the interval check logic is working correctly,')
            print('Person 1 should have been skipped in the second run.')
            print('If they were not skipped, there is a bug in the prefetch or interval logic.')

if __name__ == "__main__":
    debug_interval_check()
