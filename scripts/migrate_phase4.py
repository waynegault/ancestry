#!/usr/bin/env python3
"""
Migration script for Phase 4: Database Schema Updates.
Adds 'suggested_facts' table and updates 'conversation_state' table.
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DB_PATH = project_root / "Data" / "ancestry.db"


def migrate():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. Nothing to migrate.")
        return

    print(f"Migrating database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Create suggested_facts table
        print("Creating 'suggested_facts' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suggested_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                people_id INTEGER NOT NULL,
                fact_type VARCHAR NOT NULL,
                original_value TEXT,
                new_value TEXT NOT NULL,
                source_message_id VARCHAR,
                status VARCHAR NOT NULL,
                confidence_score INTEGER,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                FOREIGN KEY(people_id) REFERENCES people(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for suggested_facts
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_suggested_facts_people_id ON suggested_facts (people_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_suggested_facts_fact_type ON suggested_facts (fact_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_suggested_facts_status ON suggested_facts (status)")

        # 2. Update conversation_state table
        # Check if columns exist first
        cursor.execute("PRAGMA table_info(conversation_state)")
        columns = [info[1] for info in cursor.fetchall()]

        if "status" not in columns:
            print("Adding 'status' column to 'conversation_state'...")
            cursor.execute("ALTER TABLE conversation_state ADD COLUMN status VARCHAR DEFAULT 'ACTIVE'")
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_conversation_state_status ON conversation_state (status)")

        if "safety_flag" not in columns:
            print("Adding 'safety_flag' column to 'conversation_state'...")
            cursor.execute("ALTER TABLE conversation_state ADD COLUMN safety_flag BOOLEAN DEFAULT 0")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_conversation_state_safety_flag ON conversation_state (safety_flag)"
            )

        if "last_intent" not in columns:
            print("Adding 'last_intent' column to 'conversation_state'...")
            cursor.execute("ALTER TABLE conversation_state ADD COLUMN last_intent VARCHAR")

        conn.commit()
        print("Migration completed successfully.")

    except Exception as e:
        print(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
