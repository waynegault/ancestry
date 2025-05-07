#!/usr/bin/env python3
"""
Script to find people in the GEDCOM file by name and birth/death years.
Loads the GEDCOM file only once for efficiency.
"""

import sys
from gedcom_utils import GedcomData
from config import config_instance


def main():
    """Main function to find people in the GEDCOM file."""
    print("Loading GEDCOM file...")

    # Load the GEDCOM file
    gedcom_path = config_instance.GEDCOM_FILE_PATH
    if not gedcom_path:
        print("ERROR: GEDCOM_FILE_PATH not set in config.")
        return

    # Create a GedcomData instance
    try:
        gedcom_data = GedcomData(gedcom_path)
    except Exception as e:
        print(f"ERROR: Failed to load GEDCOM file: {e}")
        return

    # Define the people to search for
    people_to_find = [
        ("Margaret Thomson Simpson", 1886, 1955),
        ("Isobella Simpson", 1845, 1922),
        ("Alexander Simpson", 1840, 1878),
        ("Margaret Simpson", 1865, 1946),
        ("Alexander Stables", 1899, 1948),
        ("Catherine Margaret Stables", 1924, 2004),
        ("Frances Margaret Milne", None, None),
        ("Wayne Gordon Gault", 1969, None),
    ]

    # Find each person
    found_ids = {}
    for name_part, birth_year, death_year in people_to_find:
        print(
            f"\nSearching for '{name_part}' (Birth: {birth_year}, Death: {death_year})..."
        )

        # Search for the person
        found = []
        for norm_id, data in gedcom_data.processed_data_cache.items():
            full_name = data.get("full_name_disp", "")
            b_year = data.get("birth_year")
            d_year = data.get("death_year")

            name_match = name_part.lower() in full_name.lower()
            birth_match = birth_year is None or b_year == birth_year
            death_match = death_year is None or d_year == death_year

            if name_match and birth_match and death_match:
                found.append((norm_id, full_name, b_year, d_year))

        # Print the results
        if found:
            print(f"Found {len(found)} matching individuals:")
            for norm_id, full_name, b_year, d_year in found:
                birth_str = f"{b_year}" if b_year else "Unknown"
                death_str = f"{d_year}" if d_year else "Unknown"
                print(f"ID: {norm_id} - {full_name} ({birth_str}-{death_str})")

                # Store the first match for each person
                if name_part not in found_ids:
                    found_ids[name_part] = norm_id
        else:
            print("No matching individuals found.")

    # Print a summary of the found IDs
    print("\n=== Found IDs Summary ===")
    for name, id in found_ids.items():
        print(f"{name}: {id}")

    # Test the relationship path between Wayne Gordon Gault and Margaret Thomson Simpson
    if "Wayne Gordon Gault" in found_ids and "Margaret Thomson Simpson" in found_ids:
        wayne_id = found_ids["Wayne Gordon Gault"]
        margaret_id = found_ids["Margaret Thomson Simpson"]

        print(f"\n=== Testing Relationship Path ===")
        print(
            f"Getting relationship path between Wayne Gordon Gault ({wayne_id}) and Margaret Thomson Simpson ({margaret_id})..."
        )

        relationship_path = gedcom_data.get_relationship_path(wayne_id, margaret_id)

        print("\n=== Relationship Path ===")
        print(relationship_path)
        print("========================\n")


if __name__ == "__main__":
    main()
