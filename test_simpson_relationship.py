#!/usr/bin/env python3
"""
Test script for the relationship path between Wayne Gordon Gault and Margaret Thomson Simpson.
This script tests the specific relationship path through the Simpson family.
"""

import logging
from gedcom_utils import GedcomData
from config import config_instance

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_simpson_relationship")

def test_simpson_relationship():
    """Test the relationship path between Wayne Gordon Gault and Margaret Thomson Simpson."""
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
    isobella_simpson_id = "I102558077333"  # Isobella Simpson
    margaret_thomson_simpson_id = "I102631865823"  # Margaret Thomson Simpson
    
    # Verify each relationship individually
    print("\n=== Testing Individual Relationships ===")
    
    # Wayne -> Frances (mother)
    relationship = gedcom_data.get_relationship_path(wayne_id, frances_id)
    print(f"\nWayne -> Frances (mother):\n{relationship}")
    
    # Frances -> Catherine (mother)
    relationship = gedcom_data.get_relationship_path(frances_id, catherine_id)
    print(f"\nFrances -> Catherine (mother):\n{relationship}")
    
    # Catherine -> Alexander Stables (father)
    relationship = gedcom_data.get_relationship_path(catherine_id, alexander_stables_id)
    print(f"\nCatherine -> Alexander Stables (father):\n{relationship}")
    
    # Alexander Stables -> Margaret Simpson (mother)
    relationship = gedcom_data.get_relationship_path(alexander_stables_id, margaret_simpson_id)
    print(f"\nAlexander Stables -> Margaret Simpson (mother):\n{relationship}")
    
    # Margaret Simpson -> Alexander Simpson (father)
    relationship = gedcom_data.get_relationship_path(margaret_simpson_id, alexander_simpson_id)
    print(f"\nMargaret Simpson -> Alexander Simpson (father):\n{relationship}")
    
    # Alexander Simpson -> Isobella Simpson (sister)
    relationship = gedcom_data.get_relationship_path(alexander_simpson_id, isobella_simpson_id)
    print(f"\nAlexander Simpson -> Isobella Simpson (sister):\n{relationship}")
    
    # Isobella Simpson -> Margaret Thomson Simpson (daughter)
    relationship = gedcom_data.get_relationship_path(isobella_simpson_id, margaret_thomson_simpson_id)
    print(f"\nIsobella Simpson -> Margaret Thomson Simpson (daughter):\n{relationship}")
    
    # Now test the full relationship path
    print("\n=== Testing Full Relationship Path ===")
    relationship = gedcom_data.get_relationship_path(wayne_id, margaret_thomson_simpson_id)
    print(f"\nWayne -> Margaret Thomson Simpson (full path):\n{relationship}")
    
    # Manually construct the expected relationship path
    print("\n=== Expected Relationship Path ===")
    expected_path = """Wayne Gordon Gault (b. 1969)
  -> whose mother is Frances Margaret Milne (b. 1947)
  -> whose mother is Catherine Margaret Stables (b. 1924)
  -> whose father is Alexander Stables (b. 1899)
  -> whose mother is Margaret Simpson (b. 1865)
  -> whose father is Alexander Simpson (b. 1840)
  -> whose sister is Isobella Simpson (b. 1845)
  -> whose daughter is Margaret Thomson Simpson (b. 1886)"""
    print(expected_path)
    
    return True

if __name__ == "__main__":
    logger.info("Starting test_simpson_relationship.py")
    result = test_simpson_relationship()
    if result:
        logger.info("Test completed.")
    else:
        logger.error("Test failed.")
