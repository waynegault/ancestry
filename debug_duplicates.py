#!/usr/bin/env python3
"""
Debug script to analyze duplicate detection issues in Action 6.
"""

import sqlite3


def _connect_db(db_path: str = 'Data/ancestry.db') -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()


def _print_overview(cursor: sqlite3.Cursor) -> tuple[int, int, int]:
    print("\n📊 DATABASE OVERVIEW:")
    cursor.execute("SELECT COUNT(*) FROM people")
    total_records = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT profile_id) FROM people WHERE profile_id IS NOT NULL")
    unique_profiles = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT uuid) FROM people")
    unique_uuids = cursor.fetchone()[0]

    print(f"Total records: {total_records}")
    print(f"Unique profile_ids: {unique_profiles}")
    print(f"Unique UUIDs: {unique_uuids}")
    print(f"Profile ID duplicates: {total_records - unique_profiles}")
    return total_records, unique_profiles, unique_uuids


def _find_profile_id_duplicates(cursor: sqlite3.Cursor) -> list[tuple[str, int, str, str]]:
    print("\n🔍 PROFILE ID DUPLICATES:")
    cursor.execute(
        """
        SELECT profile_id, COUNT(*) as count,
               GROUP_CONCAT(uuid) as uuids,
               GROUP_CONCAT(username) as usernames
        FROM people
        WHERE profile_id IS NOT NULL
        GROUP BY profile_id
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT 10
        """
    )
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"Found {len(duplicates)} profile_ids with duplicates:")
        for profile_id, count, uuids, usernames in duplicates:
            print(f"  {profile_id}: {count} records")
            print(f"    UUIDs: {uuids}")
            print(f"    Users: {usernames}")
            print()
    else:
        print("No profile ID duplicates found")
    return duplicates


def _print_null_profile_analysis(cursor: sqlite3.Cursor) -> int:
    print("\n🔍 NULL PROFILE IDS:")
    cursor.execute("SELECT COUNT(*) FROM people WHERE profile_id IS NULL")
    null_profiles = cursor.fetchone()[0]
    print(f"Records with NULL profile_id: {null_profiles}")

    if null_profiles > 0:
        cursor.execute(
            """
            SELECT uuid, username, datetime(created_at) as created
            FROM people
            WHERE profile_id IS NULL
            ORDER BY created_at DESC
            LIMIT 5
            """
        )
        for uuid, username, created in cursor.fetchall():
            print(f"  {uuid[:8]}... | {username} | {created}")
    return null_profiles


def _print_recent_additions(cursor: sqlite3.Cursor) -> None:
    print("\n🔍 RECENT ADDITIONS (Last 50):")
    cursor.execute(
        """
        SELECT id, profile_id, uuid, username,
               datetime(created_at) as created,
               datetime(updated_at) as updated
        FROM people
        ORDER BY created_at DESC
        LIMIT 50
        """
    )
    recent_records = cursor.fetchall()
    creation_times: dict[str, list] = {}
    for record in recent_records:
        created_time = record[4][:16]
        creation_times.setdefault(created_time, []).append(record)

    print("Recent records grouped by creation time:")
    for time_group, records in list(creation_times.items())[:5]:
        print(f"\n  {time_group}: {len(records)} records")
        for record in records[:3]:
            id_val, profile_id, uuid, username, created, updated = record
            profile_short = (profile_id[:12] + "...") if profile_id else "NULL"
            uuid_short = (uuid[:8] + "...") if uuid else "NULL"
            print(f"    {id_val} | {profile_short} | {uuid_short} | {username}")
        if len(records) > 3:
            print(f"    ... and {len(records) - 3} more")


def _check_uuid_case_sensitivity(cursor: sqlite3.Cursor) -> list[tuple[str, int]]:
    print("\n🔍 UUID CASE SENSITIVITY CHECK:")
    cursor.execute(
        """
        SELECT uuid, COUNT(*) as count
        FROM people
        GROUP BY LOWER(uuid)
        HAVING COUNT(*) > 1
        LIMIT 5
        """
    )
    case_duplicates = cursor.fetchall()
    if case_duplicates:
        print(f"Found {len(case_duplicates)} UUIDs with case sensitivity issues:")
        for uuid, count in case_duplicates:
            print(f"  {uuid}: {count} variations")
    else:
        print("No UUID case sensitivity issues found")
    return case_duplicates


def _print_timestamp_analysis(cursor: sqlite3.Cursor) -> tuple[int, int, int, float]:
    print("\n🔍 TIMESTAMP ANALYSIS:")
    cursor.execute(
        """
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN created_at = updated_at THEN 1 END) as new_records,
            COUNT(CASE WHEN created_at != updated_at THEN 1 END) as updated_records,
            AVG(julianday(updated_at) - julianday(created_at)) * 24 * 3600 as avg_diff_seconds
        FROM people
        """
    )
    total, new_records, updated_records, avg_diff = cursor.fetchone()
    print(f"Total records: {total}")
    print(f"New records (created_at = updated_at): {new_records}")
    print(f"Updated records (created_at != updated_at): {updated_records}")
    print(f"Average time difference: {avg_diff:.3f} seconds")
    return total, new_records, updated_records, float(avg_diff or 0)


def _print_summary(duplicates, null_profiles: int, total: int, new_records: int) -> None:
    print("\n📋 ANALYSIS SUMMARY:")
    print("=" * 50)
    if duplicates:
        print("❌ ISSUE: Profile ID duplicates detected")
        print(f"   - {len(duplicates)} profile_ids have multiple records")
        print("   - This suggests UUID-based deduplication is failing")
    if null_profiles > 10:
        print("⚠️  WARNING: Many NULL profile_ids")
        print(f"   - {null_profiles} records have NULL profile_id")
        print("   - These cannot be deduplicated by profile_id")
    if new_records == total:
        print("❌ ISSUE: All records are 'new' (created_at = updated_at)")
        print("   - No records show signs of being updated")
        print("   - Script is creating new records instead of updating existing ones")
    print("\n🎯 RECOMMENDED ACTIONS:")
    if duplicates:
        print("1. Fix UUID-based duplicate detection in _lookup_existing_persons")
        print("2. Add profile_id-based duplicate detection as backup")
        print("3. Add logging to show why existing records aren't found")
    if null_profiles > 10:
        print("4. Investigate why profile_ids are NULL for many records")
        print("5. Consider alternative deduplication strategies for NULL profile_ids")


def analyze_duplicates() -> None:
    """Analyze the database for duplicate detection issues."""

    print("🔍 DUPLICATE DETECTION ANALYSIS")
    print("=" * 50)

    conn, cursor = _connect_db()

    total_records, unique_profiles, unique_uuids = _print_overview(cursor)
    duplicates = _find_profile_id_duplicates(cursor)
    null_profiles = _print_null_profile_analysis(cursor)
    _print_recent_additions(cursor)
    _check_uuid_case_sensitivity(cursor)
    total, new_records, updated_records, avg_diff = _print_timestamp_analysis(cursor)

    conn.close()

    _print_summary(duplicates, null_profiles, total, new_records)

if __name__ == "__main__":
    analyze_duplicates()
