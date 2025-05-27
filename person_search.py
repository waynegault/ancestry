# person_search.py
"""
Unified module for searching and retrieving person information from GEDCOM and Ancestry API.
Provides functions for searching, getting family details, and relationship paths.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json

# Import from local modules
from logging_config import logger
from config import config_instance
from utils import SessionManager
from gedcom_utils import GedcomData
from gedcom_search_utils import (
    search_gedcom_for_criteria,
    get_gedcom_family_details,
    get_gedcom_relationship_path,
)
from api_search_utils import (
    search_api_for_criteria,
    get_api_family_details,
    get_api_relationship_path,
)


def search_for_person(
    session_manager: Optional[SessionManager],
    search_criteria: Dict[str, Any],
    max_results: int = 10,
    search_method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for a person using both GEDCOM and API sources.

    Args:
        session_manager: SessionManager instance (required for API search)
        search_criteria: Dictionary of search criteria (first_name, surname, gender, birth_year, etc.)
        max_results: Maximum number of results to return (default: 10)
        search_method: Search method to use (GEDCOM, API, or None for both)

    Returns:
        List of dictionaries containing match information, sorted by score (highest first)
    """
    # Get search method from config if not provided
    if search_method is None:
        search_method = get_config_value("TREE_SEARCH_METHOD", "GEDCOM")
    
    # Normalize search method
    search_method = search_method.upper() if search_method else "GEDCOM"
    
    # Initialize results
    results = []
    
    # Search GEDCOM if requested
    if search_method in ["GEDCOM", "BOTH"]:
        logger.info("Searching GEDCOM data...")
        gedcom_results = search_gedcom_for_criteria(
            search_criteria=search_criteria,
            max_results=max_results,
        )
        results.extend(gedcom_results)
        logger.info(f"Found {len(gedcom_results)} matches in GEDCOM data")
    
    # Search API if requested and session manager is provided
    if search_method in ["API", "BOTH"] and session_manager:
        logger.info("Searching Ancestry API...")
        api_results = search_api_for_criteria(
            session_manager=session_manager,
            search_criteria=search_criteria,
            max_results=max_results,
        )
        results.extend(api_results)
        logger.info(f"Found {len(api_results)} matches in Ancestry API")
    
    # Sort results by score (highest first)
    results.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    # Return top results (limited by max_results)
    return results[:max_results] if results else []


def get_family_details(
    session_manager: Optional[SessionManager],
    person_id: str,
    source: str = "AUTO",
) -> Dict[str, Any]:
    """
    Get family details for a specific individual from GEDCOM or API.

    Args:
        session_manager: SessionManager instance (required for API search)
        person_id: Person ID (GEDCOM ID or Ancestry API person ID)
        source: Source to use (GEDCOM, API, or AUTO to determine from ID format)

    Returns:
        Dictionary containing family details (parents, spouses, children, siblings)
    """
    # Determine source from ID format if AUTO
    if source == "AUTO":
        # GEDCOM IDs typically start with I or @ and contain numbers
        if person_id.startswith("I") or person_id.startswith("@"):
            source = "GEDCOM"
        else:
            source = "API"
    
    # Get family details from GEDCOM
    if source == "GEDCOM":
        logger.info(f"Getting family details for {person_id} from GEDCOM")
        return get_gedcom_family_details(person_id)
    
    # Get family details from API
    elif source == "API" and session_manager:
        logger.info(f"Getting family details for {person_id} from API")
        return get_api_family_details(session_manager, person_id)
    
    # Return empty result if source is invalid or session manager is missing
    logger.error(f"Invalid source {source} or missing session manager")
    return {}


def get_relationship_path(
    session_manager: Optional[SessionManager],
    person_id: str,
    reference_id: Optional[str] = None,
    reference_name: Optional[str] = "Reference Person",
    source: str = "AUTO",
) -> str:
    """
    Get the relationship path between an individual and the reference person.

    Args:
        session_manager: SessionManager instance (required for API search)
        person_id: Person ID (GEDCOM ID or Ancestry API person ID)
        reference_id: Optional reference person ID (default: from config)
        reference_name: Optional reference person name (default: "Reference Person")
        source: Source to use (GEDCOM, API, or AUTO to determine from ID format)

    Returns:
        Formatted relationship path string
    """
    # Determine source from ID format if AUTO
    if source == "AUTO":
        # GEDCOM IDs typically start with I or @ and contain numbers
        if person_id.startswith("I") or person_id.startswith("@"):
            source = "GEDCOM"
        else:
            source = "API"
    
    # Get relationship path from GEDCOM
    if source == "GEDCOM":
        logger.info(f"Getting relationship path for {person_id} from GEDCOM")
        return get_gedcom_relationship_path(
            individual_id=person_id,
            reference_id=reference_id,
            reference_name=reference_name,
        )
    
    # Get relationship path from API
    elif source == "API" and session_manager:
        logger.info(f"Getting relationship path for {person_id} from API")
        return get_api_relationship_path(
            session_manager=session_manager,
            person_id=person_id,
            reference_id=reference_id,
            reference_name=reference_name,
        )
    
    # Return error message if source is invalid or session manager is missing
    logger.error(f"Invalid source {source} or missing session manager")
    return f"(Cannot get relationship path: {'invalid source' if source != 'API' else 'missing session manager'})"


def get_person_json(
    session_manager: Optional[SessionManager],
    person_id: str,
    source: str = "AUTO",
    include_family: bool = True,
    include_relationship: bool = True,
) -> Dict[str, Any]:
    """
    Get comprehensive JSON data for a person, including family details and relationship path.

    Args:
        session_manager: SessionManager instance (required for API search)
        person_id: Person ID (GEDCOM ID or Ancestry API person ID)
        source: Source to use (GEDCOM, API, or AUTO to determine from ID format)
        include_family: Whether to include family details (default: True)
        include_relationship: Whether to include relationship path (default: True)

    Returns:
        Dictionary containing person details, family details, and relationship path
    """
    # Determine source from ID format if AUTO
    if source == "AUTO":
        # GEDCOM IDs typically start with I or @ and contain numbers
        if person_id.startswith("I") or person_id.startswith("@"):
            source = "GEDCOM"
        else:
            source = "API"
    
    # Initialize result
    result = {
        "id": person_id,
        "source": source,
    }
    
    # Get family details if requested
    if include_family:
        family_details = get_family_details(
            session_manager=session_manager,
            person_id=person_id,
            source=source,
        )
        
        # Add person details from family details
        if family_details:
            result.update({
                "name": family_details.get("name", ""),
                "first_name": family_details.get("first_name", ""),
                "surname": family_details.get("surname", ""),
                "gender": family_details.get("gender", ""),
                "birth_year": family_details.get("birth_year"),
                "birth_date": family_details.get("birth_date", "Unknown"),
                "birth_place": family_details.get("birth_place", "Unknown"),
                "death_year": family_details.get("death_year"),
                "death_date": family_details.get("death_date", "Unknown"),
                "death_place": family_details.get("death_place", "Unknown"),
                "family": {
                    "parents": family_details.get("parents", []),
                    "siblings": family_details.get("siblings", []),
                    "spouses": family_details.get("spouses", []),
                    "children": family_details.get("children", []),
                }
            })
    
    # Get relationship path if requested
    if include_relationship:
        relationship_path = get_relationship_path(
            session_manager=session_manager,
            person_id=person_id,
            source=source,
        )
        
        if relationship_path:
            result["relationship_path"] = relationship_path
    
    return result


def get_config_value(key: str, default_value: Any = None) -> Any:
    """Safely retrieve a configuration value with fallback."""
    if not config_instance:
        return default_value
    return getattr(config_instance, key, default_value)


# Main function for testing
if __name__ == "__main__":
    import sys
    from utils import SessionManager
    
    # Create session manager
    session_manager = SessionManager()
    
    # Initialize session
    if not session_manager.start_sess("person_search.py test"):
        print("Failed to start session")
        sys.exit(1)
    
    # Test search
    search_criteria = {
        "first_name": "John",
        "surname": "Smith",
        "birth_year": 1900,
    }
    
    print(f"Searching for: {json.dumps(search_criteria)}")
    results = search_for_person(
        session_manager=session_manager,
        search_criteria=search_criteria,
        max_results=5,
    )
    
    print(f"Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.get('first_name')} {result.get('surname')} ({result.get('birth_year', '?')}-{result.get('death_year', '?')})")
    
    # Test get_person_json if results found
    if results:
        person_id = results[0]["id"]
        source = results[0]["source"]
        
        print(f"\nGetting details for {person_id} from {source}")
        person_json = get_person_json(
            session_manager=session_manager,
            person_id=person_id,
            source=source,
        )
        
        print(f"Person details: {person_json.get('name')} ({person_json.get('birth_year', '?')}-{person_json.get('death_year', '?')})")
        print(f"Family members: {len(person_json.get('family', {}).get('parents', []))} parents, {len(person_json.get('family', {}).get('siblings', []))} siblings, {len(person_json.get('family', {}).get('spouses', []))} spouses, {len(person_json.get('family', {}).get('children', []))} children")
        print(f"Relationship path: {person_json.get('relationship_path', 'None')[:100]}...")
    
    # Close session
    session_manager.close_sess()
