#!/usr/bin/env python3
"""
Migration script to update ConversationLog table schema.
Changes composite primary key to auto-incrementing ID to allow message history.
"""

import sqlite3
from datetime import datetime


def _create_backup(db_path: str, backup_path: str) -> bool:
    """Create a backup of the database."""
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print("Backup created successfully.")
        return True
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return False


def _check_migration_needed(cursor) -> bool:
    """Check if migration is needed by looking for 'id' column."""
    cursor.execute("PRAGMA table_info(conversation_log)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    return 'id' not in column_names


def _create_new_table(cursor) -> None:
    """Create the new conversation_log table with auto-incrementing ID."""
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


def _copy_data(cursor) -> int:
    """Copy data from old table to new table."""
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
    return cursor.rowcount


def _recreate_indexes(cursor) -> None:
    """Recreate all indexes for the conversation_log table."""
    indexes = [
        ("ix_conversation_log_conv_direction", "(conversation_id, direction)"),
        ("ix_conversation_log_people_id_direction_ts", "(people_id, direction, latest_timestamp)"),
        ("ix_conversation_log_timestamp", "(latest_timestamp)"),
        ("ix_conversation_log_custom_reply_sent_at", "(custom_reply_sent_at)"),
        ("ix_conversation_log_ai_sentiment", "(ai_sentiment)"),
        ("ix_conversation_log_direction", "(direction)")
    ]

    for index_name, columns in indexes:
        cursor.execute(f"CREATE INDEX {index_name} ON conversation_log {columns}")


def migrate_conversation_log() -> bool:
    """Migrate ConversationLog table to new schema with auto-incrementing ID."""
    db_path = 'data/ancestry.db'
    backup_path = f'data/ancestry_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'

    from pathlib import Path
    if not Path(db_path).exists():
        print(f"Database file {db_path} does not exist.")
        return False

    print("Starting ConversationLog migration...")
    print(f"Creating backup: {backup_path}")

    if not _create_backup(db_path, backup_path):
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not _check_migration_needed(cursor):
            print("Migration already completed - 'id' column exists.")
            conn.close()
            return True

        print("Starting schema migration...")
        _create_new_table(cursor)

        rows_copied = _copy_data(cursor)
        print(f"Copied {rows_copied} rows to new table.")

        cursor.execute("DROP TABLE conversation_log")
        cursor.execute("ALTER TABLE conversation_log_new RENAME TO conversation_log")

        _recreate_indexes(cursor)

        conn.commit()
        conn.close()

        print("Migration completed successfully!")
        print("ConversationLog table now uses auto-incrementing ID primary key.")
        print(f"Backup saved as: {backup_path}")
        return True

    except Exception as e:
        print(f"Migration failed: {e}")
        from pathlib import Path
        if Path(backup_path).exists():
            print("Restoring from backup...")
            import shutil
            shutil.copy2(backup_path, db_path)
            print("Database restored from backup.")
        return False

if __name__ == "__main__":
    success = migrate_conversation_log()
    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
