#!/usr/bin/env python3
"""
Add missing database indexes for optimal performance.

This script creates indexes that are defined in the ORM models but missing
from the actual database file.
"""

import sqlite3
import sys
from pathlib import Path

from config.config_manager import ConfigManager


def get_database_path() -> Path:
    """Get the database file path from config."""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    db_path = config.database.database_file
    if db_path is None:
        print("❌ Database file path not configured")
        sys.exit(1)
    if not db_path.exists():
        print(f"❌ Database file not found: {db_path}")
        sys.exit(1)
    return db_path


def create_missing_indexes(conn: sqlite3.Connection) -> int:
    """Create all missing indexes.

    Returns:
        Number of indexes created
    """
    cursor = conn.cursor()
    created_count = 0

    # Define all missing indexes
    indexes_to_create = [
        # conversation_log indexes
        ("ix_conversation_log_people_id", "conversation_log", "people_id"),
        ("ix_conversation_log_direction", "conversation_log", "direction"),
        ("ix_conversation_log_conversation_id", "conversation_log", "conversation_id"),
        ("ix_conversation_log_latest_timestamp", "conversation_log", "latest_timestamp"),
        ("ix_conversation_log_ai_sentiment", "conversation_log", "ai_sentiment"),
        ("ix_conversation_log_conv_direction", "conversation_log", "conversation_id, direction"),

        # dna_match indexes
        ("ix_dna_match_people_id", "dna_match", "people_id"),
        ("ix_dna_match_cm_dna", "dna_match", "cm_dna"),
    ]

    print("=" * 80)
    print("CREATING MISSING DATABASE INDEXES")
    print("=" * 80)
    print()

    for index_name, table_name, columns in indexes_to_create:
        try:
            # Check if index already exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                (index_name,)
            )
            if cursor.fetchone():
                print(f"  ⏭️  {index_name:50s} (already exists)")
                continue

            # Create the index
            sql = f"CREATE INDEX {index_name} ON {table_name} ({columns})"
            cursor.execute(sql)
            created_count += 1
            print(f"  ✅ {index_name:50s} (created)")

        except sqlite3.Error as e:
            print(f"  ❌ {index_name:50s} (error: {e})")

    conn.commit()

    print()
    print("=" * 80)
    print(f"SUMMARY: Created {created_count} new indexes")
    print("=" * 80)

    return created_count


def main():
    """Main entry point."""
    try:
        db_path = get_database_path()
        print(f"Database: {db_path}\n")

        conn = sqlite3.connect(str(db_path))
        try:
            created_count = create_missing_indexes(conn)

            if created_count > 0:
                print(f"\n✅ Successfully created {created_count} indexes")
                print("\n💡 Run verify_database_indexes.py to confirm all indexes are present")
            else:
                print("\n✅ All indexes already exist")

            sys.exit(0)

        finally:
            conn.close()

    except Exception as e:
        print(f"\n❌ Error during index creation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

