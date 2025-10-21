#!/usr/bin/env python3
"""
Migration script to add tree_statistics_cache table to existing databases.
"""

import sqlite3
from pathlib import Path


def migrate_database(db_path: str) -> None:
    """Add tree_statistics_cache table to database."""
    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='tree_statistics_cache'
    """)

    if cursor.fetchone():
        print("  ✓ tree_statistics_cache table already exists")
        conn.close()
        return

    # Create the table
    cursor.execute("""
        CREATE TABLE tree_statistics_cache (
            id INTEGER PRIMARY KEY,
            profile_id TEXT NOT NULL UNIQUE,
            total_matches INTEGER NOT NULL DEFAULT 0,
            in_tree_count INTEGER NOT NULL DEFAULT 0,
            out_tree_count INTEGER NOT NULL DEFAULT 0,
            close_matches INTEGER NOT NULL DEFAULT 0,
            moderate_matches INTEGER NOT NULL DEFAULT 0,
            distant_matches INTEGER NOT NULL DEFAULT 0,
            ethnicity_regions TEXT,
            calculated_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)

    # Create index on profile_id
    cursor.execute("""
        CREATE INDEX ix_tree_statistics_cache_profile_id
        ON tree_statistics_cache (profile_id)
    """)

    conn.commit()
    print("  ✓ tree_statistics_cache table created successfully")
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

