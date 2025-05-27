#!/usr/bin/env python3

import sqlite3
import os
from pathlib import Path

# Connect to the database
db_path = (
    Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "test_ai_responses.db"
)
print(f"Checking database at: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the database exists and has the conversation_log table
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_log'"
    )
    if not cursor.fetchone():
        print("conversation_log table not found in database")
        exit(1)

    # Count total records
    cursor.execute("SELECT COUNT(*) FROM conversation_log")
    total_count = cursor.fetchone()[0]
    print(f"Total records in conversation_log: {total_count}")

    # Count IN records
    cursor.execute("SELECT COUNT(*) FROM conversation_log WHERE direction = 'IN'")
    in_count = cursor.fetchone()[0]
    print(f"IN records: {in_count}")

    # Count OUT records
    cursor.execute("SELECT COUNT(*) FROM conversation_log WHERE direction = 'OUT'")
    out_count = cursor.fetchone()[0]
    print(f"OUT records: {out_count}")

    # Get column names
    cursor.execute("PRAGMA table_info(conversation_log)")
    columns = cursor.fetchall()
    print("\nTable columns:")
    for column in columns:
        print(f"  {column[1]} ({column[2]})")

    # Check the content of OUT records
    cursor.execute(
        "SELECT conversation_id, people_id, latest_message_content FROM conversation_log WHERE direction = 'OUT' LIMIT 5"
    )
    out_records = cursor.fetchall()

    print("\nSample OUT records:")
    for record in out_records:
        conversation_id, people_id, content = record
        print(f"Conversation ID: {conversation_id}, Person ID: {people_id}")
        print(f"Content: {content[:100]}...")
        print("-" * 50)

    conn.close()
    print("Database check complete")
except Exception as e:
    print(f"Error checking database: {e}")
