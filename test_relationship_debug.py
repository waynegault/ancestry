#!/usr/bin/env python3
"""
Debug script for the relationship path between Wayne Gordon Gault and Alexander Simpson.
This script tests the specific relationship path and prints detailed information about the path.
"""

import logging
from gedcom_utils import GedcomData, _normalize_id
from config import config_instance

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_relationship_debug")

def test_relationship_debug():
    """Test the relationship path between Wayne Gordon Gault and Alexander Simpson."""
    # Get the GEDCOM file path from the config
    gedcom_path = config_instance.GEDCOM_FILE_PATH
    if not gedcom_path:
        logger.error("GEDCOM_FILE_PATH not set in config.")
        return False
    
    # Create a GedcomData instance
    try:
        gedcom_data = GedcomData(gedcom_path)
    except Exception as e:
        logger.error(f"Failed to load GEDCOM file: {e}")
        return False
    
    # Define the IDs for all people in the relationship path
    wayne_id = "I102281560836"  # Wayne Gordon Gault
    frances_id = "I102281560544"  # Frances Margaret Milne
    catherine_id = "I102281560677"  # Catherine Margaret Stables
    alexander_stables_id = "I102281560684"  # Alexander Stables
    margaret_simpson_id = "I102281560698"  # Margaret Simpson
    alexander_simpson_id = "I102281560308"  # Alexander Simpson
    
    # Print information about each person
    print("\n=== Person Information ===")
    for person_id, name in [
        (wayne_id, "Wayne Gordon Gault"),
        (frances_id, "Frances Margaret Milne"),
        (catherine_id, "Catherine Margaret Stables"),
        (alexander_stables_id, "Alexander Stables"),
        (margaret_simpson_id, "Margaret Simpson"),
        (alexander_simpson_id, "Alexander Simpson"),
    ]:
        person = gedcom_data.find_individual_by_id(person_id)
        if person:
            print(f"\n{name} ({person_id}):")
            print(f"  Parents: {gedcom_data.id_to_parents.get(_normalize_id(person_id), set())}")
            print(f"  Children: {gedcom_data.id_to_children.get(_normalize_id(person_id), set())}")
        else:
            print(f"\n{name} ({person_id}): Not found")
    
    # Test the relationship path from Wayne to Alexander Simpson
    print("\n=== Testing Relationship Path ===")
    relationship = gedcom_data.get_relationship_path(wayne_id, alexander_simpson_id)
    print(f"\nWayne -> Alexander Simpson:\n{relationship}")
    
    # Test the relationship path from Alexander Simpson to Wayne
    print("\n=== Testing Reverse Relationship Path ===")
    relationship = gedcom_data.get_relationship_path(alexander_simpson_id, wayne_id)
    print(f"\nAlexander Simpson -> Wayne:\n{relationship}")
    
    # Manually construct the expected relationship path
    print("\n=== Expected Relationship Path ===")
    expected_path = """Wayne Gordon Gault (b. 1969)
  -> whose mother is Frances Margaret Milne (b. 1947)
  -> whose mother is Catherine Margaret Stables (b. 1924)
  -> whose father is Alexander Stables (b. 1899)
  -> whose mother is Margaret Simpson (b. 1865)
  -> whose father is Alexander Simpson (b. 1840)"""
    print(expected_path)
    
    return True

if __name__ == "__main__":
    logger.info("Starting test_relationship_debug.py")
    result = test_relationship_debug()
    if result:
        logger.info("Test completed.")
    else:
        logger.error("Test failed.")
