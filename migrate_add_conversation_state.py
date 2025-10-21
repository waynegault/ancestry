#!/usr/bin/env python3
"""
Migration script to add conversation_state table to existing databases.
Part of Phase 2: Person Lookup Integration.

This table tracks conversation state and engagement for intelligent dialogue management.
"""

import sqlite3
from pathlib import Path


def migrate_database(db_path: str) -> None:
    """Add conversation_state table to database."""
    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='conversation_state'
    """)

    if cursor.fetchone():
        print("  ✓ conversation_state table already exists")
        conn.close()
        return

    # Create the table
    cursor.execute("""
        CREATE TABLE conversation_state (
            id INTEGER PRIMARY KEY,
            people_id INTEGER NOT NULL UNIQUE,
            conversation_phase TEXT NOT NULL DEFAULT 'initial_outreach',
            engagement_score INTEGER NOT NULL DEFAULT 0,
            last_topic TEXT,
            pending_questions TEXT,
            mentioned_people TEXT,
            shared_ancestors TEXT,
            next_action TEXT,
            next_action_date TIMESTAMP,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            FOREIGN KEY (people_id) REFERENCES people (id) ON DELETE CASCADE
        )
    """)

    # Create index on people_id
    cursor.execute("""
        CREATE INDEX ix_conversation_state_people_id
        ON conversation_state (people_id)
    """)

    conn.commit()
    print("  ✓ conversation_state table created successfully")
    conn.close()


if __name__ == "__main__":
    # Migrate both production and test databases
    databases = [
        "Data/ancestry.db",
        "Data/ancestry_test.db"
    ]

    for db_path in databases:
        if Path(db_path).exists():
            migrate_database(db_path)
        else:
            print(f"  ⚠ Database not found: {db_path}")

    print("\n✓ Migration complete!")

