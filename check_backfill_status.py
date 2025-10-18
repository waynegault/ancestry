"""Quick script to check backfill status."""
from sqlalchemy import text
from core.database_manager import DatabaseManager

db_manager = DatabaseManager()

with db_manager.get_session_context() as session:
    # Total matches
    result = session.execute(text("SELECT COUNT(*) FROM dna_match")).fetchone()
    total_matches = result[0]
    
    # Matches with ethnicity data
    result = session.execute(text(
        "SELECT COUNT(*) FROM dna_match WHERE ethnicity_08302 IS NOT NULL AND ethnicity_08302 > 0"
    )).fetchone()
    matches_with_ethnicity = result[0]
    
    # Matches without ethnicity data
    matches_without_ethnicity = total_matches - matches_with_ethnicity
    
    # Sample of matches with ethnicity data
    result = session.execute(text(
        """
        SELECT dm.ethnicity_08302, dm.ethnicity_06842, dm.ethnicity_08103, dm.ethnicity_06810
        FROM dna_match dm
        WHERE dm.ethnicity_08302 IS NOT NULL AND dm.ethnicity_08302 > 0
        LIMIT 10
        """
    )).fetchall()

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
    print(f"{'08302 (84%)':<12} {'06842 (6%)':<12} {'08103 (6%)':<12} {'06810 (4%)':<12}")
    print("-" * 80)
    for row in result:
        print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12}")
    print("=" * 80)

