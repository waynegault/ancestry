#!/usr/bin/env python3
"""
Debug script to analyze duplicate detection issues in Action 6.
"""

import sqlite3
from typing import Dict, List


def analyze_duplicates():
    """Analyze the database for duplicate detection issues."""

    print("🔍 DUPLICATE DETECTION ANALYSIS")
    print("=" * 50)

    # Connect to database
    conn = sqlite3.connect('Data/ancestry.db')
    cursor = conn.cursor()

    # 1. Overall statistics
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

    # 2. Find profile ID duplicates
    print("\n🔍 PROFILE ID DUPLICATES:")
    cursor.execute("""
        SELECT profile_id, COUNT(*) as count,
               GROUP_CONCAT(uuid) as uuids,
               GROUP_CONCAT(username) as usernames
        FROM people
        WHERE profile_id IS NOT NULL
        GROUP BY profile_id
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT 10
    """)

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

    # 3. Check NULL profile_ids
    print("\n🔍 NULL PROFILE IDS:")
    cursor.execute("""
        SELECT COUNT(*) FROM people WHERE profile_id IS NULL
    """)
    null_profiles = cursor.fetchone()[0]
    print(f"Records with NULL profile_id: {null_profiles}")

    if null_profiles > 0:
        cursor.execute("""
            SELECT uuid, username, datetime(created_at) as created
            FROM people
            WHERE profile_id IS NULL
            ORDER BY created_at DESC
            LIMIT 5
        """)
        null_records = cursor.fetchall()
        print("Recent NULL profile_id records:")
        for uuid, username, created in null_records:
            print(f"  {uuid[:8]}... | {username} | {created}")

    # 4. Check recent additions
    print("\n🔍 RECENT ADDITIONS (Last 50):")
    cursor.execute("""
        SELECT id, profile_id, uuid, username,
               datetime(created_at) as created,
               datetime(updated_at) as updated
        FROM people
        ORDER BY created_at DESC
        LIMIT 50
    """)

    recent_records = cursor.fetchall()

    # Group by creation time to see batches
    creation_times: Dict[str, List] = {}
    for record in recent_records:
        created_time = record[4][:16]  # Group by minute
        if created_time not in creation_times:
            creation_times[created_time] = []
        creation_times[created_time].append(record)

    print("Recent records grouped by creation time:")
    for time_group, records in list(creation_times.items())[:5]:  # Show last 5 time groups
        print(f"\n  {time_group}: {len(records)} records")
        for record in records[:3]:  # Show first 3 records in each group
            id_val, profile_id, uuid, username, created, updated = record
            profile_short = profile_id[:12] + "..." if profile_id else "NULL"
            uuid_short = uuid[:8] + "..." if uuid else "NULL"
            print(f"    {id_val} | {profile_short} | {uuid_short} | {username}")
        if len(records) > 3:
            print(f"    ... and {len(records) - 3} more")

    # 5. Check for UUID case sensitivity issues
    print("\n🔍 UUID CASE SENSITIVITY CHECK:")
    cursor.execute("""
        SELECT uuid, COUNT(*) as count
        FROM people
        GROUP BY LOWER(uuid)
        HAVING COUNT(*) > 1
        LIMIT 5
    """)

    case_duplicates = cursor.fetchall()
    if case_duplicates:
        print(f"Found {len(case_duplicates)} UUIDs with case sensitivity issues:")
        for uuid, count in case_duplicates:
            print(f"  {uuid}: {count} variations")
    else:
        print("No UUID case sensitivity issues found")

    # 6. Check timestamp patterns
    print("\n🔍 TIMESTAMP ANALYSIS:")
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN created_at = updated_at THEN 1 END) as new_records,
            COUNT(CASE WHEN created_at != updated_at THEN 1 END) as updated_records,
            AVG(julianday(updated_at) - julianday(created_at)) * 24 * 3600 as avg_diff_seconds
        FROM people
    """)

    timestamp_stats = cursor.fetchone()
    total, new_records, updated_records, avg_diff = timestamp_stats

    print(f"Total records: {total}")
    print(f"New records (created_at = updated_at): {new_records}")
    print(f"Updated records (created_at != updated_at): {updated_records}")
    print(f"Average time difference: {avg_diff:.3f} seconds")

    conn.close()

    # 7. Analysis summary
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

if __name__ == "__main__":
    analyze_duplicates()
