"""Quick script to check backfill status."""
from sqlalchemy import text

from core.database_manager import DatabaseManager
from setup_ethnicity_tracking import load_ethnicity_metadata

# Load metadata to get current column names
metadata = load_ethnicity_metadata()
regions = metadata.get("tree_owner_regions", [])

db_manager = DatabaseManager()

with db_manager.get_session_context() as session:
    if not session:
        print("ERROR: Failed to get database session")
        exit(1)

    # Total matches
    result = session.execute(text("SELECT COUNT(*) FROM dna_match")).fetchone()
    if not result:
        print("ERROR: Failed to query database")
        exit(1)
    total_matches = result[0]

    # Build WHERE clause for matches with any ethnicity data
    where_clauses = []
    for region in regions:
        col = region["column_name"]
        where_clauses.append(f"({col} IS NOT NULL AND {col} > 0)")

    where_sql = " OR ".join(where_clauses) if where_clauses else "1=0"

    # Matches with ethnicity data
    result = session.execute(text(f"""
        SELECT COUNT(*) FROM dna_match WHERE {where_sql}
    """)).fetchone()
    if not result:
        print("ERROR: Failed to query matches with ethnicity")
        exit(1)
    matches_with_ethnicity = result[0]

    # Matches with all zeros (processed but no overlap)
    where_all_zero = " AND ".join([f"{region['column_name']} = 0" for region in regions])
    result = session.execute(text(f"""
        SELECT COUNT(*) FROM dna_match WHERE {where_all_zero}
    """)).fetchone()
    if not result:
        print("ERROR: Failed to query matches with all zeros")
        exit(1)
    matches_all_zero = result[0]

    # Matches with all NULL (never processed)
    where_all_null = " AND ".join([f"{region['column_name']} IS NULL" for region in regions])
    result = session.execute(text(f"""
        SELECT COUNT(*) FROM dna_match WHERE {where_all_null}
    """)).fetchone()
    if not result:
        print("ERROR: Failed to query matches with all NULL")
        exit(1)
    matches_all_null = result[0]

    # Sample of matches with ethnicity data
    select_cols = ", ".join([region["column_name"] for region in regions])
    result = session.execute(text(f"""
        SELECT {select_cols}
        FROM dna_match
        WHERE {where_sql}
        LIMIT 10
    """)).fetchall()

    print("=" * 80)
    print("ETHNICITY BACKFILL STATUS")
    print("=" * 80)
    print(f"Total DNA matches: {total_matches}")
    print()
    print(f"Matches with shared ethnicity (>0% in your regions): {matches_with_ethnicity} ({matches_with_ethnicity / total_matches * 100:.1f}%)")
    print(f"Matches with NO shared ethnicity (0% in your regions): {matches_all_zero} ({matches_all_zero / total_matches * 100:.1f}%)")
    print(f"Matches not yet processed: {matches_all_null} ({matches_all_null / total_matches * 100:.1f}%)")
    print()
    print(f"Total processed: {matches_with_ethnicity + matches_all_zero} ({(matches_with_ethnicity + matches_all_zero) / total_matches * 100:.1f}%)")
    print("=" * 80)
    print("\nSample of matches with ethnicity data (first 10):")
    print("-" * 80)

    # Print header with region names
    header = ""
    for region in regions:
        name = region['name'][:18]  # Truncate long names
        header += f"{name:<20} "
    print(header)
    print("-" * 80)

    for row in result:
        row_str = ""
        for val in row:
            row_str += f"{val or 0:<20} "
        print(row_str)
    print("=" * 80)

