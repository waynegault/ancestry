# action10.py: GEDCOM Report (local fuzzy search, family, relationship to WGG)
import sys
import os
from pathlib import Path
from logging_config import setup_logging
from config import config_instance
from utils import SessionManager

# Import all relevant helpers from utils
from utils import (
    GEDCOM_LIB_AVAILABLE,
    GedcomReader,
    build_indi_index,
    _is_name,
    _normalize_id,
    _get_full_name,
    find_individual_by_id,
    find_potential_matches,
    display_gedcom_family_details,
    get_relationship_path,
    extract_and_fix_id,
)
import utils


def run_action10():
    if not GEDCOM_LIB_AVAILABLE or GedcomReader is None:
        print("ged4py library unavailable. Cannot run GEDCOM report.")
        return
    gedcom_path_str = getattr(config_instance, "GEDCOM_FILE_PATH", None)
    if not gedcom_path_str:
        print("GEDCOM_FILE_PATH not set in config.")
        return
    gedcom_path = Path(gedcom_path_str)
    if not gedcom_path.is_file():
        print(f"GEDCOM not found: {gedcom_path}")
        return
    reader = GedcomReader(str(gedcom_path))
    build_indi_index(reader)
    wgg_search_name_lower = "wayne gordon gault"
    wayne_gault_indi = None
    wayne_gault_id_gedcom = None
    for indi in utils.INDI_INDEX.values():
        name_rec = indi.name
        name_str = name_rec.format() if _is_name(name_rec) else ""
        if name_str.lower() == wgg_search_name_lower:
            wayne_gault_indi = indi
            wayne_gault_id_gedcom = _normalize_id(indi.xref_id)
            break
    if not wayne_gault_indi or not wayne_gault_id_gedcom:
        print(
            "Wayne Gordon Gault (reference person) not found in local GEDCOM. Cannot calculate relationships."
        )
        return
    print("\nEnter as many details as you know. Leave blank to skip a field.")
    first_name = input("First name: ").strip() or None
    surname = input("Surname (or maiden name): ").strip() or None
    dob_str = input("Date of birth (YYYY-MM-DD or year): ").strip() or None
    pob = input("Place of birth: ").strip() or None
    gender = input("Gender (M/F, optional): ").strip().lower() or None
    if gender and gender not in ("m", "f"):
        gender = None
    dod_str = input("Date of death (YYYY-MM-DD or year, optional): ").strip() or None
    pod = input("Place of death (optional): ").strip() or None
    matches = find_potential_matches(
        reader, first_name, surname, dob_str, pob, dod_str, pod, gender
    )
    if not matches:
        print("\nNo matches found in GEDCOM.")
        return
    print(f"\nFound {len(matches)} potential matches:")
    for i, match in enumerate(matches):
        print(f"  {i+1}. {match['name']}")
        print(f"     Born : {match['birth_date']} in {match['birth_place']}")
        if (match["death_date"] and match["death_date"] != "N/A") or (
            match["death_place"] and match["death_place"] != "N/A"
        ):
            print(f"     Died : {match['death_date']} in {match['death_place']}")
        print(f"     Reasons: {match['reasons']}")
    if len(matches) == 1 or (
        len(matches) > 1 and matches[0]["score"] != matches[1]["score"]
    ):
        selected_match = matches[0]
        print(f"\nAuto-selected: {selected_match['name']}")
    else:
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(matches):
                print("Selection cancelled or invalid.")
                return
            selected_match = matches[choice - 1]
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            return
    selected_id = extract_and_fix_id(selected_match["id"])
    if not selected_id:
        print("ERROR: Invalid ID in selected match.")
        return
    selected_indi = find_individual_by_id(reader, selected_id)
    if not selected_indi:
        print("ERROR: Could not retrieve individual record from GEDCOM.")
        return
    print("\n=== PERSON DETAILS ===")
    print(f"Name: {selected_match['name']}")
    print(f"Gender: {gender.upper() if gender else 'N/A'}")
    print(f"Birth : {selected_match['birth_date']} in {selected_match['birth_place']}")
    if (selected_match["death_date"] and selected_match["death_date"] != "N/A") or (
        selected_match["death_place"] and selected_match["death_place"] != "N/A"
    ):
        print(
            f"Death : {selected_match['death_date']} in {selected_match['death_place']}"
        )
    tree_id = os.getenv("MY_TREE_ID")
    ancestry_id = (
        selected_id[1:]
        if selected_id
        and selected_id[0] in ("I", "F", "S", "T", "N", "M", "C", "X", "O")
        else selected_id
    )
    if tree_id and ancestry_id:
        print(
            f"Link in Tree: https://www.ancestry.co.uk/family-tree/person/tree/{tree_id}/person/{ancestry_id}/facts"
        )
    else:
        print(f"Link in Tree: (unavailable)")
    display_gedcom_family_details(reader, selected_indi)
    relationship_path = get_relationship_path(
        reader, selected_id, wayne_gault_id_gedcom
    )
    print("\nRelationship Path:\n")
    print(relationship_path.strip())


if __name__ == "__main__":
    run_action10()
