#!/usr/bin/env python3
"""
Migration script to update ConversationLog table schema.
Changes composite primary key to auto-incrementing ID to allow message history.
"""

import os
import sqlite3
from datetime import datetime


def migrate_conversation_log():
    """Migrate ConversationLog table to new schema with auto-incrementing ID."""

    db_path = 'data/ancestry.db'
    backup_path = f'data/ancestry_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'

    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return False

    print("Starting ConversationLog migration...")
    print(f"Creating backup: {backup_path}")

    try:
        # Create backup
        import shutil
        shutil.copy2(db_path, backup_path)
        print("Backup created successfully.")

        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if migration is needed
        cursor.execute("PRAGMA table_info(conversation_log)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        if 'id' in column_names:
            print("Migration already completed - 'id' column exists.")
            conn.close()
            return True

        print("Starting schema migration...")

        # Step 1: Create new table with correct schema
        cursor.execute("""
            CREATE TABLE conversation_log_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id VARCHAR NOT NULL,
                direction VARCHAR(3) NOT NULL,
                people_id INTEGER NOT NULL,
                latest_message_content TEXT,
                latest_timestamp DATETIME NOT NULL,
                ai_sentiment VARCHAR,
                message_template_id INTEGER,
                script_message_status VARCHAR,
                updated_at DATETIME NOT NULL,
                custom_reply_sent_at DATETIME,
                FOREIGN KEY(people_id) REFERENCES people (id),
                FOREIGN KEY(message_template_id) REFERENCES message_templates (id)
            )
        """)

        # Step 2: Copy data from old table to new table
        cursor.execute("""
            INSERT INTO conversation_log_new (
                conversation_id, direction, people_id, latest_message_content,
                latest_timestamp, ai_sentiment, message_template_id,
                script_message_status, updated_at, custom_reply_sent_at
            )
            SELECT
                conversation_id, direction, people_id, latest_message_content,
                latest_timestamp, ai_sentiment, message_template_id,
                script_message_status, updated_at, custom_reply_sent_at
            FROM conversation_log
        """)

        rows_copied = cursor.rowcount
        print(f"Copied {rows_copied} rows to new table.")

        # Step 3: Drop old table
        cursor.execute("DROP TABLE conversation_log")

        # Step 4: Rename new table
        cursor.execute("ALTER TABLE conversation_log_new RENAME TO conversation_log")

        # Step 5: Recreate indexes
        cursor.execute("""
            CREATE INDEX ix_conversation_log_conv_direction
            ON conversation_log (conversation_id, direction)
        """)

        cursor.execute("""
            CREATE INDEX ix_conversation_log_people_id_direction_ts
            ON conversation_log (people_id, direction, latest_timestamp)
        """)

        cursor.execute("""
            CREATE INDEX ix_conversation_log_timestamp
            ON conversation_log (latest_timestamp)
        """)

        cursor.execute("""
            CREATE INDEX ix_conversation_log_custom_reply_sent_at
            ON conversation_log (custom_reply_sent_at)
        """)

        cursor.execute("""
            CREATE INDEX ix_conversation_log_ai_sentiment
            ON conversation_log (ai_sentiment)
        """)

        cursor.execute("""
            CREATE INDEX ix_conversation_log_direction
            ON conversation_log (direction)
        """)

        # Commit changes
        conn.commit()
        conn.close()

        print("Migration completed successfully!")
        print("ConversationLog table now uses auto-incrementing ID primary key.")
        print(f"Backup saved as: {backup_path}")

        return True

    except Exception as e:
        print(f"Migration failed: {e}")
        if os.path.exists(backup_path):
            print("Restoring from backup...")
            shutil.copy2(backup_path, db_path)
            print("Database restored from backup.")
        return False

if __name__ == "__main__":
    success = migrate_conversation_log()
    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
