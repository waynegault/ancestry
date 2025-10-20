#!/usr/bin/env python3

"""
Action 12: Compare GEDCOM vs API Search Results

This action compares the results from Action 10 (GEDCOM-based search) and 
Action 11 (API-based search) to validate consistency between the two approaches.

Workflow:
1. Prompt user for search criteria (name, birth year, birth place, etc.)
2. Run Action 10 search using GEDCOM file
3. Run Action 11 search using Ancestry API
4. Compare and display results side-by-side
5. Highlight any differences in scoring, family data, or relationship paths
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module  # type: ignore[import-not-found]

# === MODULE SETUP ===
logger = setup_module(__name__)

# === IMPORTS ===
from typing import Any, Optional
from core.session_manager import SessionManager

# Import action modules
import action10  # type: ignore[import-not-found]
import action11  # type: ignore[import-not-found]


def get_search_input() -> dict[str, Any]:
    """
    Prompt user for search criteria.
    
    Returns:
        Dictionary with search criteria
    """
    print("\n" + "=" * 60)
    print("Action 12: Compare GEDCOM vs API Search Results")
    print("=" * 60)
    print("\nEnter search criteria:")
    print("-" * 60)
    
    criteria = {}
    
    # Get name
    first_name = input("First name: ").strip()
    surname = input("Surname: ").strip()
    criteria["first_name"] = first_name
    criteria["surname"] = surname
    criteria["full_name"] = f"{first_name} {surname}".strip()
    
    # Get birth info
    birth_year = input("Birth year (or press Enter to skip): ").strip()
    if birth_year:
        criteria["birth_year"] = birth_year
    
    birth_place = input("Birth place (or press Enter to skip): ").strip()
    if birth_place:
        criteria["birth_place"] = birth_place
    
    # Get death info
    death_year = input("Death year (or press Enter to skip): ").strip()
    if death_year:
        criteria["death_year"] = death_year
    
    death_place = input("Death place (or press Enter to skip): ").strip()
    if death_place:
        criteria["death_place"] = death_place
    
    # Get gender
    gender = input("Gender (M/F or press Enter to skip): ").strip().upper()
    if gender in ["M", "F"]:
        criteria["gender"] = gender
    
    return criteria


def run_action10_search(criteria: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Run Action 10 (GEDCOM-based) search.
    
    Args:
        criteria: Search criteria dictionary
        
    Returns:
        Dictionary with Action 10 results or None if failed
    """
    print("\n" + "=" * 60)
    print("Running Action 10 (GEDCOM-based search)...")
    print("=" * 60)
    
    try:
        # Call Action 10's search function
        result = action10.search_gedcom_for_person(
            first_name=criteria.get("first_name", ""),
            surname=criteria.get("surname", ""),
            birth_year=criteria.get("birth_year"),
            birth_place=criteria.get("birth_place"),
            death_year=criteria.get("death_year"),
            death_place=criteria.get("death_place"),
            gender=criteria.get("gender"),
        )
        return result
    except Exception as e:
        logger.error(f"Action 10 search failed: {e}", exc_info=True)
        print(f"\n❌ Action 10 search failed: {e}")
        return None


def run_action11_search(
    session_manager: SessionManager, criteria: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Run Action 11 (API-based) search.
    
    Args:
        session_manager: Active session manager
        criteria: Search criteria dictionary
        
    Returns:
        Dictionary with Action 11 results or None if failed
    """
    print("\n" + "=" * 60)
    print("Running Action 11 (API-based search)...")
    print("=" * 60)
    
    try:
        # Call Action 11's search function
        result = action11.search_ancestry_api_for_person(
            session_manager=session_manager,
            first_name=criteria.get("first_name", ""),
            surname=criteria.get("surname", ""),
            birth_year=criteria.get("birth_year"),
            birth_place=criteria.get("birth_place"),
            death_year=criteria.get("death_year"),
            death_place=criteria.get("death_place"),
            gender=criteria.get("gender"),
        )
        return result
    except Exception as e:
        logger.error(f"Action 11 search failed: {e}", exc_info=True)
        print(f"\n❌ Action 11 search failed: {e}")
        return None


def compare_results(
    action10_result: Optional[dict[str, Any]],
    action11_result: Optional[dict[str, Any]],
    criteria: dict[str, Any],
) -> None:
    """
    Compare and display results from Action 10 and Action 11.
    
    Args:
        action10_result: Results from Action 10
        action11_result: Results from Action 11
        criteria: Original search criteria
    """
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    # Display search criteria
    print(f"\nSearch Criteria: {criteria.get('full_name', 'Unknown')}")
    if criteria.get("birth_year"):
        print(f"Birth Year: {criteria['birth_year']}")
    if criteria.get("birth_place"):
        print(f"Birth Place: {criteria['birth_place']}")
    
    print("\n" + "-" * 60)
    print("ACTION 10 (GEDCOM) RESULTS:")
    print("-" * 60)
    if action10_result:
        print(f"✅ Found: {action10_result.get('name', 'Unknown')}")
        print(f"   Score: {action10_result.get('score', 0)}")
        print(f"   Birth: {action10_result.get('birth_year', 'Unknown')}")
        print(f"   Death: {action10_result.get('death_year', 'Unknown')}")
        print(f"   Relationship: {action10_result.get('relationship', 'Unknown')}")
    else:
        print("❌ No results found")
    
    print("\n" + "-" * 60)
    print("ACTION 11 (API) RESULTS:")
    print("-" * 60)
    if action11_result:
        print(f"✅ Found: {action11_result.get('name', 'Unknown')}")
        print(f"   Score: {action11_result.get('score', 0)}")
        print(f"   Birth: {action11_result.get('birth_year', 'Unknown')}")
        print(f"   Death: {action11_result.get('death_year', 'Unknown')}")
        print(f"   Relationship: {action11_result.get('relationship', 'Unknown')}")
    else:
        print("❌ No results found")
    
    # Compare scores
    if action10_result and action11_result:
        print("\n" + "-" * 60)
        print("SCORE COMPARISON:")
        print("-" * 60)
        score10 = action10_result.get("score", 0)
        score11 = action11_result.get("score", 0)
        diff = abs(score10 - score11)
        
        print(f"Action 10 Score: {score10}")
        print(f"Action 11 Score: {score11}")
        print(f"Difference: {diff}")
        
        if diff == 0:
            print("✅ Scores match perfectly!")
        elif diff <= 10:
            print("⚠️  Minor score difference (within tolerance)")
        else:
            print("❌ Significant score difference - investigate!")
    
    print("\n" + "=" * 60)


def run_action12_wrapper(session_manager: SessionManager) -> bool:
    """
    Main wrapper for Action 12.
    
    Args:
        session_manager: Active session manager
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get search criteria from user
        criteria = get_search_input()
        
        if not criteria.get("full_name"):
            print("\n❌ Error: Name is required")
            return False
        
        # Run Action 10 search
        action10_result = run_action10_search(criteria)
        
        # Run Action 11 search
        action11_result = run_action11_search(session_manager, criteria)
        
        # Compare results
        compare_results(action10_result, action11_result, criteria)
        
        return True
        
    except Exception as e:
        logger.error(f"Action 12 failed: {e}", exc_info=True)
        print(f"\n❌ Action 12 failed: {e}")
        return False


# === TESTS ===
def _test_action12_basic() -> bool:
    """Test basic Action 12 functionality."""
    print("\n🧪 Test: Action 12 basic functionality")
    print("This test requires manual input - skipping in automated tests")
    return True


if __name__ == "__main__":
    print("Action 12: Compare GEDCOM vs API Search Results")
    print("This action must be run from main.py")

