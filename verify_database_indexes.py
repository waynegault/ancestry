#!/usr/bin/env python3
"""
Verify and report on database indexes for optimal performance.

This utility checks that all required indexes exist for efficient querying,
especially important when scaling to 15,000+ conversations.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def get_existing_indexes(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, str]]]:
    """Get all existing indexes grouped by table."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT tbl_name, name, sql
        FROM sqlite_master
        WHERE type='index'
        AND tbl_name IN ('people', 'conversation_log', 'dna_match', 'family_tree', 'message_templates')
        ORDER BY tbl_name, name
    """)

    indexes_by_table: Dict[str, List[Tuple[str, str]]] = {}
    for table, index_name, sql in cursor.fetchall():
        if table not in indexes_by_table:
            indexes_by_table[table] = []
        indexes_by_table[table].append((index_name, sql or "PRIMARY KEY"))

    return indexes_by_table


def get_required_indexes() -> Dict[str, List[Tuple[str, str, str]]]:
    """Define required indexes for optimal performance.

    Returns:
        Dict mapping table name to list of (index_name, columns, purpose) tuples
    """
    return {
        "people": [
            ("ix_people_profile_id", "profile_id", "Person lookup by profile ID (Action 7 prefetch)"),
            ("ix_people_uuid", "uuid", "Person lookup by DNA test GUID"),
            ("ix_people_status", "status", "Filter by person status (ACTIVE, DESIST, etc.)"),
            ("ix_people_in_my_tree", "in_my_tree", "Filter by tree membership"),
            ("ix_people_last_logged_in", "last_logged_in", "Sort by last login date"),
            ("ix_people_deleted_at", "deleted_at", "Soft delete filtering"),
            ("ix_people_administrator_profile_id", "administrator_profile_id", "Kit administrator lookup"),
        ],
        "conversation_log": [
            ("ix_conversation_log_people_id", "people_id", "Lookup logs by person"),
            ("ix_conversation_log_direction", "direction", "Filter by message direction (IN/OUT)"),
            ("ix_conversation_log_conversation_id", "conversation_id", "Lookup by conversation ID (Action 7 prefetch)"),
            ("ix_conversation_log_latest_timestamp", "latest_timestamp", "Sort by message timestamp (smart skip logic)"),
            ("ix_conversation_log_ai_sentiment", "ai_sentiment", "Filter by AI classification"),
            ("ix_conversation_log_custom_reply_sent_at", "custom_reply_sent_at", "Track custom replies"),
            ("ix_conversation_log_conv_direction", "conversation_id, direction", "Composite: conversation + direction"),
            ("ix_conversation_log_people_id_direction_ts", "people_id, direction, latest_timestamp", "Composite: person + direction + time"),
            ("ix_conversation_log_timestamp", "latest_timestamp", "Timestamp-only queries"),
        ],
        "dna_match": [
            ("ix_dna_match_people_id", "people_id", "Link to person record"),
            ("ix_dna_match_cm_dna", "cm_dna", "Sort/filter by shared DNA amount"),
        ],
        "family_tree": [
            ("ix_family_tree_people_id", "people_id", "Link to person record"),
            ("ix_family_tree_cfpid", "cfpid", "Lookup by Ancestry person ID"),
            ("ix_family_tree_actual_relationship", "actual_relationship", "Filter by relationship type"),
        ],
        "message_templates": [
            ("ix_message_templates_template_key", "template_key", "Lookup template by key"),
            ("ix_message_templates_template_category", "template_category", "Filter by category"),
            ("ix_message_templates_tree_status", "tree_status", "Filter by tree status"),
            ("ix_message_templates_is_active", "is_active", "Filter active templates"),
        ],
    }


def verify_indexes() -> Tuple[int, int, List[str]]:
    """Verify all required indexes exist.

    Returns:
        Tuple of (total_required, total_found, missing_indexes)
    """
    db_path = get_database_path()
    conn = sqlite3.connect(str(db_path))

    try:
        existing_indexes = get_existing_indexes(conn)
        required_indexes = get_required_indexes()

        total_required = sum(len(indexes) for indexes in required_indexes.values())
        total_found = 0
        missing_indexes = []

        print("=" * 80)
        print("DATABASE INDEX VERIFICATION REPORT")
        print("=" * 80)
        print(f"Database: {db_path}")
        print()

        for table, required in required_indexes.items():
            print(f"\n📋 Table: {table}")
            print("-" * 80)

            existing = existing_indexes.get(table, [])
            existing_names = {name for name, _ in existing}

            for index_name, columns, purpose in required:
                if index_name in existing_names:
                    print(f"  ✅ {index_name:50s} ({columns})")
                    total_found += 1
                else:
                    print(f"  ❌ {index_name:50s} ({columns}) - MISSING")
                    print(f"     Purpose: {purpose}")
                    missing_indexes.append(f"{table}.{index_name}")

        print("\n" + "=" * 80)
        print(f"SUMMARY: {total_found}/{total_required} required indexes found")

        if missing_indexes:
            print(f"\n⚠️  {len(missing_indexes)} MISSING INDEXES:")
            for idx in missing_indexes:
                print(f"   - {idx}")
            print("\n💡 Run database schema migration to create missing indexes")
        else:
            print("\n✅ All required indexes are present!")

        print("=" * 80)

        return total_required, total_found, missing_indexes

    finally:
        conn.close()


def main():
    """Main entry point."""
    try:
        _, _, missing_indexes = verify_indexes()

        # Exit with error code if indexes are missing
        if missing_indexes:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during index verification: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

