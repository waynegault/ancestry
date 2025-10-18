"""Quick script to check backfill status."""
from sqlalchemy import text
from core.database_manager import DatabaseManager
from setup_ethnicity_tracking import load_ethnicity_metadata

# Load metadata to get current column names
metadata = load_ethnicity_metadata()
regions = metadata.get("tree_owner_regions", [])

db_manager = DatabaseManager()

with db_manager.get_session_context() as session:
    # Total matches
    result = session.execute(text("SELECT COUNT(*) FROM dna_match")).fetchone()
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
    matches_with_ethnicity = result[0]

    # Matches without ethnicity data
    matches_without_ethnicity = total_matches - matches_with_ethnicity

    # Sample of matches with ethnicity data
    select_cols = ", ".join([region["column_name"] for region in regions])
    result = session.execute(text(f"""
        SELECT {select_cols}
        FROM dna_match
        WHERE {where_sql}
        LIMIT 10
    """)).fetchall()

    print("=" * 80)
    print("BACKFILL STATUS")
    print("=" * 80)
    print(f"Total DNA matches: {total_matches}")
    print(f"Matches with ethnicity data: {matches_with_ethnicity}")
    print(f"Matches without ethnicity data: {matches_without_ethnicity}")
    if total_matches > 0:
        print(f"Completion: {matches_with_ethnicity / total_matches * 100:.1f}%")
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

