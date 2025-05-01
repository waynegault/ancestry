# action10.py
"""
Identifies candidate matches in a GEDCOM file, scores them, retrieves family details,
and displays the relationship path to the tree owner.
"""

import logging
from typing import Dict, List, Optional
from config import config_instance
from gedcom_utils import GedcomData, GedcomIndividualType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("action10")


def get_user_input() -> Dict[str, Optional[str]]:
    """Collects user input for search criteria."""
    print("\n--- Enter Search Criteria ---")
    first_name = input("First Name: ").strip() or None
    surname = input("Surname: ").strip() or None
    dob = input("Birth Date (e.g., 15 Feb 2001): ").strip() or None
    pob = input("Birth Place: ").strip() or None
    dod = input("Death Date (optional): ").strip() or None
    pod = input("Death Place (optional): ").strip() or None
    gender = input("Gender (M/F, optional): ").strip().upper() or None
    gender = gender[0] if gender in ("M", "F") else None
    return {
        "first_name": first_name,
        "surname": surname,
        "dob": dob,
        "pob": pob,
        "dod": dod,
        "pod": pod,
        "gender": gender,
    }


def display_candidate(candidate: Dict) -> None:
    """Displays candidate information."""
    print("\n--- Top Candidate ---")
    print(f"Name: {candidate['name']}")
    print(f"Gender: {candidate.get('gender', 'Unknown')}")
    print(
        f"Birth: {candidate['birth_date']} in {candidate.get('birth_place', 'Unknown')}"
    )
    if candidate.get("death_date", "N/A") != "N/A":
        print(
            f"Death: {candidate['death_date']} in {candidate.get('death_place', 'Unknown')}"
        )
    print(f"Match Score: {candidate['score']}")
    print(f"Match Reasons: {candidate['reasons']}")


def display_family(gedcom: GedcomData, individual: GedcomIndividualType) -> None:
    """Retrieves and displays family members of the candidate."""
    print("\n--- Family Members ---")
    relationships = ["parents", "siblings", "spouses", "children"]
    for rel in relationships:
        members = gedcom.get_related_individuals(individual, rel)
        print(f"\n{rel.title()}:")
        if not members:
            print("  None found")
            continue
        for member in members:
            name = GedcomData._get_full_name(member)
            life_dates = GedcomData.format_life_dates(member)
            print(f"  - {name}{life_dates}")


def main():
    # Load GEDCOM file
    if not config_instance or not config_instance.GEDCOM_FILE_PATH:
        logger.error("GEDCOM path not configured in .env")
        return
    gedcom_path = config_instance.GEDCOM_FILE_PATH
    try:
        gedcom = GedcomData(gedcom_path)
    except Exception as e:
        logger.error(f"Failed to load GEDCOM: {e}")
        return

    # Get search criteria
    criteria = get_user_input()
    if not criteria["first_name"] and not criteria["surname"]:
        logger.error("At least a first name or surname is required")
        return

    # Find potential matches
    matches = gedcom.find_potential_matches(
        first_name=criteria["first_name"],
        surname=criteria["surname"],
        dob_str=criteria["dob"],
        pob=criteria["pob"],
        dod_str=criteria["dod"],
        pod=criteria["pod"],
        gender=criteria["gender"],
        scoring_weights=config_instance.COMMON_SCORING_WEIGHTS,
        name_flexibility=config_instance.NAME_FLEXIBILITY,
        date_flexibility=config_instance.DATE_FLEXIBILITY,
        max_results=10,
    )

    if not matches:
        print("\nNo matching candidates found")
        return

    # Get top candidate
    top_candidate = matches[0]
    display_candidate(top_candidate)

    # Retrieve individual object
    individual = gedcom.find_individual_by_id(top_candidate["norm_id"])
    if not individual:
        logger.error("Could not retrieve individual details")
        return

    # Display family
    display_family(gedcom, individual)

    # Get relationship path to tree owner
    tree_owner_id = config_instance.REFERENCE_PERSON_ID
    if not tree_owner_id:
        logger.warning("Tree owner ID not configured")
        return

    print("\n--- Relationship Path to Tree Owner ---")
    path_info = gedcom.get_relationship_path(top_candidate["norm_id"], tree_owner_id)
    print(path_info)


if __name__ == "__main__":
    main()
