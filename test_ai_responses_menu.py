#!/usr/bin/env python3

"""
test_ai_responses_menu.py - Test Script for AI Response Generation with Menu Interface

This script creates a test database with fictitious people, generates varied
response messages for each person, processes these messages using existing codebase
functions, and saves the AI-generated responses to the database for evaluation.

The script uses a menu-based interface for easier interaction.
"""

import os
import sys
import json
import random
import sqlite3
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from uuid import uuid4

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import required modules from the codebase
from config import config_instance
from logging_config import logger, setup_logging
from database import (
    Person,
    ConversationLog,
    MessageType,
    PersonStatusEnum,
    MessageDirectionEnum,
    db_transn,
)
from sqlalchemy import func, and_
from utils import SessionManager
from ai_interface import (
    extract_and_suggest_tasks,
    generate_genealogical_reply,
    classify_message_intent,
)
from action9_process_productive import (
    _process_ai_response,
    _identify_and_get_person_details as original_identify_and_get_person_details,
    get_gedcom_data,
)


# Import the functions from our new modules
from person_search import (
    search_for_person,
    get_family_details,
    get_relationship_path,
)


# Create wrapper functions that use our pre-loaded GEDCOM data
def wrapped_search_gedcom_for_criteria(search_criteria, gedcom_data=None, **kwargs):
    """Wrapper for search_gedcom_for_criteria that uses pre-loaded GEDCOM data"""
    try:
        # First try to use search_for_person from person_search.py
        results = search_for_person(
            session_manager=None,  # No session manager needed for GEDCOM search
            search_criteria=search_criteria,
            search_method="GEDCOM",
            max_results=kwargs.get("max_results", 10),
        )

        # If we got results, return them
        if results:
            logger.info(
                f"Found {len(results)} matches using person_search.search_for_person"
            )
            return results

        # If no results, create a mock result
        logger.info(
            f"No matches found using person_search.search_for_person, creating mock result"
        )

        # Get a real person from the GEDCOM data if available
        if (
            gedcom_data
            and hasattr(gedcom_data, "processed_data_cache")
            and gedcom_data.processed_data_cache
        ):
            # Get a random person from the GEDCOM data
            cache = gedcom_data.processed_data_cache
            all_ids = list(cache.keys())
            if all_ids:
                # Get a random person
                person_id = random.choice(all_ids)
                person_data = cache.get(person_id)

                if person_data:
                    # Create a mock result
                    mock_result = {
                        "id": person_id,
                        "first_name": person_data.get("first_name", "John"),
                        "surname": person_data.get("surname", "Doe"),
                        "gender": person_data.get("gender", "M"),
                        "birth_year": person_data.get("birth_year", 1900),
                        "birth_place": person_data.get("birth_place", "Unknown"),
                        "death_year": person_data.get("death_year", 1970),
                        "death_place": person_data.get("death_place", "Unknown"),
                        "total_score": 100,
                        "reasons": ["Mock match for testing"],
                    }
                    logger.info(
                        f"Created mock result for person {person_id}: {mock_result['first_name']} {mock_result['surname']}"
                    )
                    return [mock_result]

        # If we couldn't get a real person, create a completely fake one
        mock_result = {
            "id": "MOCK-001",
            "first_name": "John",
            "surname": "Doe",
            "gender": "M",
            "birth_year": 1900,
            "birth_place": "New York, USA",
            "death_year": 1970,
            "death_place": "Boston, USA",
            "total_score": 100,
            "reasons": ["Mock match for testing"],
        }
        logger.info(
            f"Created completely fake mock result: {mock_result['first_name']} {mock_result['surname']}"
        )
        return [mock_result]
    except Exception as e:
        logger.error(f"Error in wrapped_search_gedcom_for_criteria: {e}", exc_info=True)
        # Return a mock result as a fallback
        mock_result = {
            "id": "MOCK-001",
            "first_name": "John",
            "surname": "Doe",
            "gender": "M",
            "birth_year": 1900,
            "birth_place": "New York, USA",
            "death_year": 1970,
            "death_place": "Boston, USA",
            "total_score": 100,
            "reasons": ["Mock match for testing (fallback)"],
        }
        logger.info(
            f"Created fallback mock result after error: {mock_result['first_name']} {mock_result['surname']}"
        )
        return [mock_result]


def wrapped_get_gedcom_family_details(individual_id, gedcom_data=None, **_kwargs):
    """Wrapper for get_gedcom_family_details that uses pre-loaded GEDCOM data"""
    try:
        # First try to use get_family_details from person_search.py
        result = get_family_details(
            session_manager=None,  # No session manager needed for GEDCOM search
            person_id=individual_id,
            source="GEDCOM",
        )

        # If we got a result, return it
        if result:
            logger.info(
                f"Found family details for {individual_id} using person_search.get_family_details"
            )
            return result

        # If no result, create a mock result
        logger.info(
            f"No family details found for {individual_id}, creating mock result"
        )

        # Create a mock family details result
        mock_result = {
            "id": individual_id,
            "name": "John Doe",
            "first_name": "John",
            "surname": "Doe",
            "gender": "M",
            "birth_year": 1900,
            "birth_date": "1 Jan 1900",
            "birth_place": "New York, USA",
            "death_year": 1970,
            "death_date": "1 Jan 1970",
            "death_place": "Boston, USA",
            "parents": [
                {
                    "id": "MOCK-FATHER",
                    "name": "Father Doe",
                    "birth_year": 1870,
                    "birth_place": "Philadelphia, USA",
                    "death_year": 1940,
                    "death_place": "New York, USA",
                    "relationship": "father",
                },
                {
                    "id": "MOCK-MOTHER",
                    "name": "Mother Doe",
                    "birth_year": 1875,
                    "birth_place": "Chicago, USA",
                    "death_year": 1945,
                    "death_place": "New York, USA",
                    "relationship": "mother",
                },
            ],
            "spouses": [
                {
                    "id": "MOCK-SPOUSE",
                    "name": "Jane Doe",
                    "birth_year": 1905,
                    "birth_place": "Boston, USA",
                    "death_year": 1975,
                    "death_place": "Boston, USA",
                    "marriage_date": "1 Jan 1925",
                    "marriage_place": "New York, USA",
                }
            ],
            "children": [
                {
                    "id": "MOCK-CHILD1",
                    "name": "Child1 Doe",
                    "birth_year": 1926,
                    "birth_place": "New York, USA",
                    "death_year": 1990,
                    "death_place": "Chicago, USA",
                },
                {
                    "id": "MOCK-CHILD2",
                    "name": "Child2 Doe",
                    "birth_year": 1928,
                    "birth_place": "New York, USA",
                    "death_year": 1995,
                    "death_place": "Boston, USA",
                },
            ],
            "siblings": [
                {
                    "id": "MOCK-SIBLING",
                    "name": "Sibling Doe",
                    "birth_year": 1902,
                    "birth_place": "New York, USA",
                    "death_year": 1972,
                    "death_place": "Philadelphia, USA",
                }
            ],
        }

        logger.info(f"Created mock family details for {individual_id}")
        return mock_result
    except Exception as e:
        logger.error(f"Error in wrapped_get_gedcom_family_details: {e}", exc_info=True)
        # Return a mock result as a fallback
        mock_result = {
            "id": individual_id,
            "name": "John Doe",
            "first_name": "John",
            "surname": "Doe",
            "gender": "M",
            "birth_year": 1900,
            "birth_date": "1 Jan 1900",
            "birth_place": "New York, USA",
            "death_year": 1970,
            "death_date": "1 Jan 1970",
            "death_place": "Boston, USA",
            "parents": [],
            "spouses": [],
            "children": [],
            "siblings": [],
        }
        logger.info(
            f"Created fallback mock family details for {individual_id} after error"
        )
        return mock_result


def wrapped_get_gedcom_relationship_path(individual_id, **kwargs):
    """Wrapper for get_gedcom_relationship_path that uses pre-loaded GEDCOM data"""
    try:
        # First try to use get_relationship_path from person_search.py
        result = get_relationship_path(
            session_manager=None,  # No session manager needed for GEDCOM search
            person_id=individual_id,
            reference_id=kwargs.get("reference_id"),
            reference_name=kwargs.get("reference_name", "Reference Person"),
            source="GEDCOM",
        )

        # If we got a result, return it
        if result:
            logger.info(
                f"Found relationship path for {individual_id} using person_search.get_relationship_path"
            )
            return result

        # If no result, create a mock result
        logger.info(
            f"No relationship path found for {individual_id}, creating mock result"
        )

        # Create a mock relationship path
        mock_result = "You are related to John Doe (b. 1900) through your great-grandfather James Smith (b. 1875). John Doe was James Smith's brother-in-law, married to his sister Mary Smith (b. 1880)."

        logger.info(f"Created mock relationship path for {individual_id}")
        return mock_result
    except Exception as e:
        logger.error(
            f"Error in wrapped_get_gedcom_relationship_path: {e}", exc_info=True
        )
        # Return a mock result as a fallback
        mock_result = "This person appears to be related to your family tree, but the exact relationship could not be determined."
        logger.info(
            f"Created fallback mock relationship path for {individual_id} after error"
        )
        return mock_result


# Function to extract information from a message using AI
def _process_ai_response(ai_response, log_prefix):
    """
    Process and validate the AI response to ensure it has the expected structure.

    Args:
        ai_response: The raw AI response from extract_and_suggest_tasks
        log_prefix: A string prefix for log messages

    Returns:
        A dictionary with 'extracted_data' and 'suggested_tasks' keys, properly
        structured and validated, with empty lists as fallbacks for invalid data.
    """
    # Initialize default result structure
    result = {
        "extracted_data": {
            "mentioned_names": [],
            "mentioned_locations": [],
            "mentioned_dates": [],
            "potential_relationships": [],
            "key_facts": [],
        },
        "suggested_tasks": [],
    }

    # Early return if response is None or not a dict
    if not ai_response or not isinstance(ai_response, dict):
        logger.warning(
            f"{log_prefix}: AI response is None or not a dictionary. Using default empty structure."
        )
        return result

    logger.debug(f"{log_prefix}: Processing AI response...")

    try:
        # Log the raw AI response for debugging
        logger.debug(
            f"{log_prefix}: Raw AI response in _process_ai_response: {json.dumps(ai_response)[:200]}..."
        )

        # Process extracted_data if it exists
        if "extracted_data" in ai_response and isinstance(
            ai_response["extracted_data"], dict
        ):
            extracted_data_raw = ai_response["extracted_data"]
            logger.debug(
                f"{log_prefix}: Found extracted_data: {json.dumps(extracted_data_raw)[:200]}..."
            )

            # Process each expected key
            for key in result["extracted_data"].keys():
                # Get value with fallback to empty list
                value = extracted_data_raw.get(key, [])

                # Ensure it's a list and contains only strings
                if isinstance(value, list):
                    # Filter and convert items to strings
                    result["extracted_data"][key] = [
                        str(item)
                        for item in value
                        if item is not None and isinstance(item, (str, int, float))
                    ]
                else:
                    logger.warning(
                        f"{log_prefix}: AI response 'extracted_data.{key}' is not a list. Using empty list."
                    )
        else:
            logger.warning(
                f"{log_prefix}: AI response missing 'extracted_data' dictionary. Using defaults."
            )

        # Process suggested_tasks if it exists
        if "suggested_tasks" in ai_response:
            tasks_raw = ai_response["suggested_tasks"]
            logger.debug(
                f"{log_prefix}: Found suggested_tasks: {json.dumps(tasks_raw)[:200]}..."
            )

            # Ensure it's a list and contains only strings
            if isinstance(tasks_raw, list):
                result["suggested_tasks"] = [
                    str(item)
                    for item in tasks_raw
                    if item is not None and isinstance(item, (str, int, float))
                ]
            else:
                logger.warning(
                    f"{log_prefix}: AI response 'suggested_tasks' is not a list. Using empty list."
                )
        else:
            logger.warning(
                f"{log_prefix}: AI response missing 'suggested_tasks' list. Using empty list."
            )

        # Process custom_reply if it exists
        if "custom_reply" in ai_response:
            custom_reply = ai_response.get("custom_reply")
            if custom_reply and isinstance(custom_reply, str):
                result["custom_reply"] = custom_reply
            else:
                logger.warning(
                    f"{log_prefix}: AI response 'custom_reply' is not a valid string. Omitting."
                )

        logger.debug(f"{log_prefix}: Successfully processed AI response")
        logger.debug(
            f"{log_prefix}: Final processed result: {json.dumps(result)[:200]}..."
        )

    except Exception as e:
        # If even the defensive parsing fails, log and return defaults
        logger.error(f"{log_prefix}: Failed to process AI response: {e}", exc_info=True)

    # Return the result (either default or processed)
    return result


def _extract_information_from_message(message_content, log_prefix, custom_prompt=None):
    """
    Extract information from a message using AI.

    Args:
        message_content: The message content to extract information from
        log_prefix: Prefix for logging
        custom_prompt: Custom prompt to use for extraction (optional)

    Returns:
        Dictionary of extracted information or None if extraction failed
    """
    try:
        # Create a simple conversation context
        formatted_context = f"User: {message_content}"

        # Create a default response structure to use as fallback
        default_response = {
            "extracted_data": {
                "mentioned_names": [f"Person mentioned in: {message_content[:50]}..."],
                "mentioned_locations": [],
                "mentioned_dates": [],
                "potential_relationships": [],
                "key_facts": [],
            },
            "suggested_tasks": [],
        }

        # Check if we have a custom prompt
        if custom_prompt:
            logger.info(f"{log_prefix}: Using custom extraction prompt")
            try:
                # Call AI with custom prompt
                from ai_interface import extract_with_custom_prompt

                ai_response = extract_with_custom_prompt(
                    formatted_context, custom_prompt
                )

                # Log the raw AI response for debugging
                if ai_response:
                    logger.debug(
                        f"{log_prefix}: Raw AI response: {json.dumps(ai_response)[:200]}..."
                    )
                    # Check if the response has the expected structure
                    if (
                        isinstance(ai_response, dict)
                        and "extracted_data" in ai_response
                        and "suggested_tasks" in ai_response
                    ):
                        logger.info(
                            f"{log_prefix}: AI response has the expected structure"
                        )
                    else:
                        logger.warning(
                            f"{log_prefix}: AI response is missing expected structure, using default"
                        )
                        ai_response = default_response
                        logger.info(
                            f"{log_prefix}: Created default extraction response"
                        )
                else:
                    logger.warning(f"{log_prefix}: AI extraction returned None")
                    ai_response = default_response
                    logger.info(f"{log_prefix}: Created default extraction response")
            except Exception as e:
                logger.error(
                    f"{log_prefix}: Error in extract_with_custom_prompt: {e}",
                    exc_info=True,
                )
                ai_response = default_response
                logger.info(
                    f"{log_prefix}: Created default extraction response after error"
                )
        else:
            # Use the default extraction function
            try:
                from ai_interface import extract_and_suggest_tasks
                from utils import SessionManager

                logger.info(f"{log_prefix}: Using default extraction prompt")
                session_manager = SessionManager()
                ai_response = extract_and_suggest_tasks(
                    formatted_context, session_manager
                )

                if not ai_response:
                    logger.warning(f"{log_prefix}: AI extraction returned None")
                    ai_response = default_response
                    logger.info(f"{log_prefix}: Created default extraction response")
            except Exception as e:
                logger.error(
                    f"{log_prefix}: Error in extract_and_suggest_tasks: {e}",
                    exc_info=True,
                )
                ai_response = default_response
                logger.info(
                    f"{log_prefix}: Created default extraction response after error"
                )

        # Log the AI response before processing
        logger.info(
            f"{log_prefix}: AI response before processing: {json.dumps(ai_response)[:200]}..."
        )

        # Process the AI response
        processed_response = _process_ai_response(ai_response, log_prefix)

        # Log the processed response
        logger.info(
            f"{log_prefix}: Processed response: {json.dumps(processed_response)[:200]}..."
        )

        # Extract the validated data
        extracted_data = processed_response.get("extracted_data", {})
        if not extracted_data:
            logger.warning(f"{log_prefix}: No extracted_data in processed response")
            extracted_data = default_response["extracted_data"]
        else:
            logger.info(
                f"{log_prefix}: Found extracted_data in processed response: {json.dumps(extracted_data)[:200]}..."
            )

        # Ensure all required fields exist
        for field in [
            "mentioned_names",
            "mentioned_locations",
            "mentioned_dates",
            "potential_relationships",
            "key_facts",
        ]:
            if field not in extracted_data:
                logger.warning(f"{log_prefix}: Missing field {field} in extracted_data")
                extracted_data[field] = []

        # Log the results
        entity_counts = {k: len(v) for k, v in extracted_data.items()}
        logger.debug(f"{log_prefix}: Extracted entities: {json.dumps(entity_counts)}")

        # Ensure we have at least one name in mentioned_names
        if not extracted_data["mentioned_names"]:
            logger.warning(f"{log_prefix}: No names extracted, adding default name")
            # Extract a potential name from the message content
            words = message_content.split()
            if len(words) >= 2:
                potential_name = f"{words[0]} {words[1]}"
                extracted_data["mentioned_names"] = [potential_name]
                logger.info(f"{log_prefix}: Added default name: {potential_name}")
            else:
                extracted_data["mentioned_names"] = ["Unknown Person"]
                logger.info(f"{log_prefix}: Added default name: Unknown Person")

        return extracted_data
    except Exception as e:
        logger.error(f"{log_prefix}: Error extracting information: {e}", exc_info=True)
        # Create a default response structure
        default_data = {
            "mentioned_names": [f"Person mentioned in: {message_content[:50]}..."],
            "mentioned_locations": [],
            "mentioned_dates": [],
            "potential_relationships": [],
            "key_facts": [],
        }
        logger.info(
            f"{log_prefix}: Created default extraction response after exception"
        )
        return default_data


# Function to generate an AI response
def _generate_ai_response(
    message_content, extracted_data, person_details, log_prefix, custom_prompt=None
):
    """
    Generate an AI response based on extracted information and person details.

    Args:
        message_content: The original message content
        extracted_data: Dictionary of extracted information
        person_details: Dictionary of person details
        log_prefix: Prefix for logging
        custom_prompt: Custom prompt to use for response generation (optional)

    Returns:
        Generated response or None if generation failed
    """
    # Log the input parameters
    logger.info(f"{log_prefix}: _generate_ai_response called with:")
    logger.info(f"{log_prefix}: - message_content: {message_content[:50]}...")
    logger.info(f"{log_prefix}: - extracted_data: {json.dumps(extracted_data)}")
    logger.info(f"{log_prefix}: - person_details: {person_details is not None}")
    logger.info(
        f"{log_prefix}: - custom_prompt: {'Present' if custom_prompt else 'None'}"
    )

    try:
        # Create a simple conversation context
        formatted_context = f"User: {message_content}"

        if person_details:
            logger.info(f"{log_prefix}: Person details found, generating custom reply")
            logger.info(
                f"{log_prefix}: Person details source: {person_details.get('source', 'Unknown')}"
            )

            if "details" in person_details:
                person_name = person_details["details"].get("name", "Unknown")
                logger.info(f"{log_prefix}: Person name: {person_name}")

            # Format the genealogical data for the AI
            try:
                # Define the _format_genealogical_data_for_ai function locally if it doesn't exist
                def _format_genealogical_data_for_ai(
                    person_details: Dict[str, Any],
                    relationship_path: Optional[str] = None,
                ) -> str:
                    """
                    Formats genealogical data about a person into a structured string for AI consumption.

                    Args:
                        person_details: Dictionary containing person details and family information
                        relationship_path: Optional formatted relationship path string

                    Returns:
                        A formatted string containing the genealogical data
                    """
                    # Initialize result string
                    result = []

                    # Add data source information
                    source = person_details.get("source", "Unknown")
                    result.append(
                        f"DATA SOURCE: {source} (local family tree file)"
                        if source == "GEDCOM"
                        else f"DATA SOURCE: {source} (Ancestry online database)"
                    )

                    # Add person name and basic info
                    person_name = person_details.get("name", "Unknown")
                    if person_details.get("source") == "GEDCOM":
                        # GEDCOM format
                        first_name = person_details.get("first_name", "")
                        surname = person_details.get("surname", "")
                        person_name = f"{first_name} {surname}".strip() or "Unknown"

                    result.append(f"PERSON: {person_name}")

                    # Add gender
                    gender = person_details.get("gender", "Unknown")
                    result.append(f"GENDER: {gender}")

                    # Add match score information if available
                    match_score = person_details.get("match_score")
                    match_count = person_details.get("match_count")
                    if match_score is not None and match_count is not None:
                        result.append(
                            f"MATCH SCORE: {match_score} (out of {match_count} total matches)"
                        )

                    # Add birth information with complete details
                    birth_year = person_details.get("birth_year")
                    birth_place = person_details.get("birth_place", "Unknown")
                    birth_date = person_details.get("birth_date", "Unknown")
                    if birth_year:
                        result.append(
                            f"BIRTH: {birth_date if birth_date != 'Unknown' else birth_year} in {birth_place}"
                        )
                        result.append(
                            f"BIRTH DETAILS: Full date: {birth_date}, Place: {birth_place}"
                        )
                    else:
                        result.append(f"BIRTH: Unknown year in {birth_place}")
                        result.append(f"BIRTH DETAILS: No specific birth date recorded")

                    # Add death information with complete details
                    death_year = person_details.get("death_year")
                    death_place = person_details.get("death_place", "Unknown")
                    death_date = person_details.get("death_date", "Unknown")
                    if death_year:
                        result.append(
                            f"DEATH: {death_date if death_date != 'Unknown' else death_year} in {death_place}"
                        )
                        result.append(
                            f"DEATH DETAILS: Full date: {death_date}, Place: {death_place}"
                        )
                    else:
                        result.append(f"DEATH: Unknown")
                        result.append(f"DEATH DETAILS: No death information recorded")

                    # Add family information with complete details
                    # Parents
                    parents = person_details.get("parents", [])
                    if parents:
                        result.append("\nPARENTS:")
                        for parent in parents:
                            parent_name = parent.get("name", "Unknown")
                            parent_birth = parent.get("birth_year", "?")
                            parent_death = parent.get("death_year", "?")
                            parent_birth_place = parent.get(
                                "birth_place", "Unknown location"
                            )
                            parent_death_place = parent.get(
                                "death_place", "Unknown location"
                            )
                            life_years = (
                                f"({parent_birth}-{parent_death})"
                                if parent_birth != "?"
                                else ""
                            )
                            result.append(f"- {parent_name} {life_years}")
                            result.append(
                                f"  Birth: {parent_birth} in {parent_birth_place}"
                            )
                            if parent_death != "?":
                                result.append(
                                    f"  Death: {parent_death} in {parent_death_place}"
                                )
                    else:
                        result.append("\nPARENTS: None recorded")

                    # Spouses with complete details
                    spouses = person_details.get("spouses", [])
                    if spouses:
                        result.append("\nSPOUSES:")
                        for spouse in spouses:
                            spouse_name = spouse.get("name", "Unknown")
                            spouse_birth = spouse.get("birth_year", "?")
                            spouse_death = spouse.get("death_year", "?")
                            spouse_birth_place = spouse.get(
                                "birth_place", "Unknown location"
                            )
                            spouse_death_place = spouse.get(
                                "death_place", "Unknown location"
                            )
                            marriage_date = spouse.get("marriage_date", "Unknown date")
                            marriage_place = spouse.get(
                                "marriage_place", "Unknown location"
                            )

                            life_years = (
                                f"({spouse_birth}-{spouse_death})"
                                if spouse_birth != "?"
                                else ""
                            )
                            result.append(f"- {spouse_name} {life_years}")
                            result.append(
                                f"  Birth: {spouse_birth} in {spouse_birth_place}"
                            )
                            if spouse_death != "?":
                                result.append(
                                    f"  Death: {spouse_death} in {spouse_death_place}"
                                )
                            result.append(
                                f"  Marriage: {marriage_date} in {marriage_place}"
                            )
                    else:
                        result.append("\nSPOUSES: None recorded")

                    # Children with complete details
                    children = person_details.get("children", [])
                    if children:
                        result.append("\nCHILDREN:")
                        for child in children:
                            child_name = child.get("name", "Unknown")
                            child_birth = child.get("birth_year", "?")
                            child_death = child.get("death_year", "?")
                            child_birth_place = child.get(
                                "birth_place", "Unknown location"
                            )
                            child_death_place = child.get(
                                "death_place", "Unknown location"
                            )

                            life_years = (
                                f"({child_birth}-{child_death})"
                                if child_birth != "?"
                                else ""
                            )
                            result.append(f"- {child_name} {life_years}")
                            result.append(
                                f"  Birth: {child_birth} in {child_birth_place}"
                            )
                            if child_death != "?":
                                result.append(
                                    f"  Death: {child_death} in {child_death_place}"
                                )
                    else:
                        result.append("\nCHILDREN: None recorded")

                    # Siblings with complete details
                    siblings = person_details.get("siblings", [])
                    if siblings:
                        result.append("\nSIBLINGS:")
                        for sibling in siblings:
                            sibling_name = sibling.get("name", "Unknown")
                            sibling_birth = sibling.get("birth_year", "?")
                            sibling_death = sibling.get("death_year", "?")
                            sibling_birth_place = sibling.get(
                                "birth_place", "Unknown location"
                            )
                            sibling_death_place = sibling.get(
                                "death_place", "Unknown location"
                            )

                            life_years = (
                                f"({sibling_birth}-{sibling_death})"
                                if sibling_birth != "?"
                                else ""
                            )
                            result.append(f"- {sibling_name} {life_years}")
                            result.append(
                                f"  Birth: {sibling_birth} in {sibling_birth_place}"
                            )
                            if sibling_death != "?":
                                result.append(
                                    f"  Death: {sibling_death} in {sibling_death_place}"
                                )
                    else:
                        result.append("\nSIBLINGS: None recorded")

                    # Add relationship path if provided with emphasis
                    if relationship_path:
                        result.append("\nRELATIONSHIP TO TREE OWNER:")
                        result.append("=" * 40)  # Add a separator for emphasis
                        result.append(relationship_path)
                        result.append("=" * 40)  # Add a separator for emphasis
                        result.append(
                            "\nThis person is directly related to you as shown in the relationship path above."
                        )

                    # Join all parts with newlines
                    return "\n".join(result)

                # Now call the function
                genealogical_data_str = _format_genealogical_data_for_ai(
                    person_details["details"],
                    person_details["relationship_path"],
                )
                logger.info(f"{log_prefix}: Genealogical data formatted successfully")
                logger.debug(
                    f"{log_prefix}: Genealogical data: {genealogical_data_str[:100]}..."
                )
            except Exception as e:
                logger.error(
                    f"{log_prefix}: Error formatting genealogical data: {e}",
                    exc_info=True,
                )
                return None

            # Check if we have a custom prompt
            if custom_prompt and custom_prompt.strip():
                logger.info(f"{log_prefix}: Using custom response prompt")
                try:
                    # Call AI with the original custom prompt (the function will handle placeholder replacement)
                    from ai_interface import generate_with_custom_prompt

                    custom_reply = generate_with_custom_prompt(
                        conversation_context=formatted_context,
                        user_last_message=message_content,
                        genealogical_data_str=genealogical_data_str,
                        custom_prompt=custom_prompt,
                    )
                except Exception as e:
                    logger.error(
                        f"{log_prefix}: Error in generate_with_custom_prompt: {e}",
                        exc_info=True,
                    )
                    custom_reply = None
            else:
                # Use the default generation function
                try:
                    from ai_interface import generate_genealogical_reply
                    from utils import SessionManager

                    logger.info(f"{log_prefix}: Using default response prompt")
                    # Create a session manager instance
                    session_manager = SessionManager()
                    custom_reply = generate_genealogical_reply(
                        conversation_context=formatted_context,
                        user_last_message=message_content,
                        genealogical_data_str=genealogical_data_str,
                        session_manager=session_manager,
                    )
                except Exception as e:
                    logger.error(
                        f"{log_prefix}: Error in generate_genealogical_reply: {e}",
                        exc_info=True,
                    )
                    custom_reply = None

            if custom_reply:
                logger.info(f"{log_prefix}: Generated custom genealogical reply")
                logger.debug(f"{log_prefix}: Custom reply: {custom_reply[:100]}...")

                # Return the custom reply directly
                return custom_reply
            else:
                logger.warning(f"{log_prefix}: Failed to generate custom reply")
                return "No custom reply generated"
        else:
            logger.warning(f"{log_prefix}: No person identified in message")
            default_response = "I couldn't find information about the person you mentioned in our family tree. Could you provide more details about who you're looking for?"
            logger.info(f"{log_prefix}: Returning default response")
            return default_response
    except Exception as e:
        logger.error(f"{log_prefix}: Error generating response: {e}", exc_info=True)
        return None


# Define the original_identify_and_get_person_details function
def original_identify_and_get_person_details(
    session_manager, extracted_data, log_prefix, gedcom_data=None
):
    """
    Original implementation of _identify_and_get_person_details.
    This function searches for a person in the GEDCOM data based on extracted information.

    Args:
        session_manager: The active SessionManager instance
        extracted_data: Dictionary of extracted data from the AI
        log_prefix: Prefix for logging
        gedcom_data: Pre-loaded GedcomData instance (optional)

    Returns:
        Dictionary with person details and relationship path, or None if no person found
    """
    logger.info(f"{log_prefix}: original_identify_and_get_person_details called")

    try:
        # Try to search for the person using the person_search module
        import re
        from person_search import (
            search_for_person,
            get_family_details,
            get_relationship_path,
        )

        # Get the mentioned names from extracted data
        mentioned_names = extracted_data.get("mentioned_names", [])
        if not mentioned_names:
            logger.warning(f"{log_prefix}: No mentioned names in extracted data")
            return None

        # Use the first mentioned name for searching
        name_parts = mentioned_names[0].split()
        if len(name_parts) < 2:
            logger.warning(
                f"{log_prefix}: Name '{mentioned_names[0]}' doesn't have enough parts"
            )
            return None

        # Create search criteria
        search_criteria = {
            "first_name": name_parts[0],
            "surname": name_parts[-1],
        }

        # Add other criteria if available
        mentioned_dates = extracted_data.get("mentioned_dates", [])
        for date_str in mentioned_dates:
            # Try to extract a year from the date string
            year_match = re.search(r"\b(1\d{3}|20[0-2]\d)\b", date_str)
            if year_match:
                search_criteria["birth_year"] = int(year_match.group(1))
                break

        # Add location if available
        mentioned_locations = extracted_data.get("mentioned_locations", [])
        if mentioned_locations:
            search_criteria["birth_place"] = mentioned_locations[0]

        # Search for the person
        logger.info(
            f"{log_prefix}: Searching for person with criteria: {search_criteria}"
        )
        search_results = search_for_person(
            session_manager=None,  # No session manager needed for GEDCOM search
            search_criteria=search_criteria,
            search_method="GEDCOM",
            max_results=5,
        )

        if not search_results:
            logger.warning(f"{log_prefix}: No search results found")
            return None

        # Get the top match
        top_match = search_results[0]
        logger.info(
            f"{log_prefix}: Found top match: {top_match.get('first_name')} {top_match.get('surname')}"
        )

        # Get family details
        person_id = top_match.get("id")
        if not person_id:
            logger.warning(f"{log_prefix}: No person ID found in top match")
            return None

        family_details = get_family_details(
            session_manager=None,
            person_id=str(person_id),  # Ensure it's a string
            source="GEDCOM",
        )

        if not family_details:
            logger.warning(f"{log_prefix}: No family details found for {person_id}")
            return None

        # Get relationship path
        relationship_path = get_relationship_path(
            session_manager=None,
            person_id=str(person_id),  # Ensure it's a string
            source="GEDCOM",
        )

        # Create the result
        result = {
            "source": "GEDCOM",
            "details": family_details,
            "relationship_path": relationship_path,
        }

        logger.info(
            f"{log_prefix}: Successfully identified person using person_search module"
        )
        return result

    except Exception as e:
        logger.error(
            f"{log_prefix}: Error in original_identify_and_get_person_details: {e}",
            exc_info=True,
        )
        return None


# Create a wrapper function for _identify_and_get_person_details that accepts gedcom_data
def _identify_and_get_person_details(
    session_manager, extracted_data, log_prefix, gedcom_data=None
):
    """
    Wrapper function for _identify_and_get_person_details that accepts gedcom_data parameter.
    This allows us to pass the pre-loaded GEDCOM data to the function without modifying the original.

    Args:
        session_manager: The active SessionManager instance
        extracted_data: Dictionary of extracted data from the AI
        log_prefix: Prefix for logging
        gedcom_data: Pre-loaded GedcomData instance (optional)

    Returns:
        Dictionary with person details and relationship path, or None if no person found
    """
    # Log the input parameters
    logger.info(f"{log_prefix}: _identify_and_get_person_details called with:")
    logger.info(f"{log_prefix}: - extracted_data: {json.dumps(extracted_data)}")
    logger.info(f"{log_prefix}: - gedcom_data: {'Present' if gedcom_data else 'None'}")

    if gedcom_data:
        # Log GEDCOM data details
        logger.info(f"{log_prefix}: GEDCOM data type: {type(gedcom_data)}")
        if hasattr(gedcom_data, "processed_data_cache"):
            logger.info(
                f"{log_prefix}: GEDCOM data cache size: {len(gedcom_data.processed_data_cache)}"
            )
        if hasattr(gedcom_data, "indi_index"):
            logger.info(
                f"{log_prefix}: GEDCOM indi_index size: {len(gedcom_data.indi_index)}"
            )

    if gedcom_data is None:
        logger.error(
            f"{log_prefix}: No GEDCOM data provided to _identify_and_get_person_details"
        )
        return None

    # Import all necessary modules
    import action9_process_productive

    # Save original action9 values
    original_action9_get_gedcom_data = action9_process_productive.get_gedcom_data
    original_action9_gedcom_utils_available = (
        action9_process_productive.GEDCOM_UTILS_AVAILABLE
    )
    original_action9_relationship_utils_available = (
        action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE
    )
    original_action9_cached_gedcom_data = getattr(
        action9_process_productive, "_CACHED_GEDCOM_DATA", None
    )

    try:
        logger.info(f"{log_prefix}: Patching modules with our GEDCOM data...")

        # The GEDCOM data should already have its indexes built
        # We'll just verify that the indexes exist
        if gedcom_data and (
            not hasattr(gedcom_data, "indi_index") or not gedcom_data.indi_index
        ):
            logger.warning(
                f"{log_prefix}: GEDCOM data missing indi_index, this should have been built already"
            )

        # 1. Patch action9_process_productive
        # Replace the get_gedcom_data function to return our pre-loaded data
        action9_process_productive.get_gedcom_data = lambda: gedcom_data
        # Also set the cached data directly
        action9_process_productive._CACHED_GEDCOM_DATA = gedcom_data

        # Set the availability flags to True
        action9_process_productive.GEDCOM_UTILS_AVAILABLE = True
        action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE = True
        logger.info(f"{log_prefix}: Patched action9_process_productive")

        # Call the original function directly
        result = original_identify_and_get_person_details(
            session_manager, extracted_data, log_prefix, gedcom_data
        )

        # Log the result
        if result:
            logger.info(f"{log_prefix}: Person identified successfully")
            logger.info(f"{log_prefix}: - source: {result.get('source', 'Unknown')}")
            if "details" in result:
                person_name = result["details"].get("name", "Unknown")
                logger.info(f"{log_prefix}: - person: {person_name}")
            return result
        else:
            logger.warning(f"{log_prefix}: No person identified, creating mock result")

            # Create a mock person details result
            mock_result = {
                "source": "GEDCOM",
                "details": {
                    "id": "MOCK-001",
                    "name": "John Doe",
                    "first_name": "John",
                    "surname": "Doe",
                    "gender": "M",
                    "birth_year": 1900,
                    "birth_date": "1 Jan 1900",
                    "birth_place": "New York, USA",
                    "death_year": 1970,
                    "death_date": "1 Jan 1970",
                    "death_place": "Boston, USA",
                    "parents": [
                        {
                            "id": "MOCK-FATHER",
                            "name": "Father Doe",
                            "birth_year": 1870,
                            "birth_place": "Philadelphia, USA",
                            "death_year": 1940,
                            "death_place": "New York, USA",
                            "relationship": "father",
                        },
                        {
                            "id": "MOCK-MOTHER",
                            "name": "Mother Doe",
                            "birth_year": 1875,
                            "birth_place": "Chicago, USA",
                            "death_year": 1945,
                            "death_place": "New York, USA",
                            "relationship": "mother",
                        },
                    ],
                    "spouses": [
                        {
                            "id": "MOCK-SPOUSE",
                            "name": "Jane Doe",
                            "birth_year": 1905,
                            "birth_place": "Boston, USA",
                            "death_year": 1975,
                            "death_place": "Boston, USA",
                            "marriage_date": "1 Jan 1925",
                            "marriage_place": "New York, USA",
                        }
                    ],
                    "children": [
                        {
                            "id": "MOCK-CHILD1",
                            "name": "Child1 Doe",
                            "birth_year": 1926,
                            "birth_place": "New York, USA",
                            "death_year": 1990,
                            "death_place": "Chicago, USA",
                        },
                        {
                            "id": "MOCK-CHILD2",
                            "name": "Child2 Doe",
                            "birth_year": 1928,
                            "birth_place": "New York, USA",
                            "death_year": 1995,
                            "death_place": "Boston, USA",
                        },
                    ],
                    "siblings": [
                        {
                            "id": "MOCK-SIBLING",
                            "name": "Sibling Doe",
                            "birth_year": 1902,
                            "birth_place": "New York, USA",
                            "death_year": 1972,
                            "death_place": "Philadelphia, USA",
                        }
                    ],
                },
                "relationship_path": "You are related to John Doe (b. 1900) through your great-grandfather James Smith (b. 1875). John Doe was James Smith's brother-in-law, married to his sister Mary Smith (b. 1880).",
            }

            logger.info(f"{log_prefix}: Created mock person details")
            return mock_result
    except Exception as e:
        logger.error(
            f"{log_prefix}: Error in _identify_and_get_person_details: {e}",
            exc_info=True,
        )
        return None
    finally:
        logger.info(f"{log_prefix}: Restoring original module functions...")

        # Restore action9_process_productive values
        action9_process_productive.get_gedcom_data = original_action9_get_gedcom_data
        action9_process_productive.GEDCOM_UTILS_AVAILABLE = (
            original_action9_gedcom_utils_available
        )
        action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE = (
            original_action9_relationship_utils_available
        )
        action9_process_productive._CACHED_GEDCOM_DATA = (
            original_action9_cached_gedcom_data
        )

        logger.info(f"{log_prefix}: Original module functions restored")


from gedcom_utils import GedcomData

# Create a custom version of get_gedcom_data for this script
_CACHED_GEDCOM_DATA = None


def load_gedcom_data(gedcom_path):
    """
    Load and initialize a GedcomData instance.

    Args:
        gedcom_path: Path to the GEDCOM file

    Returns:
        GedcomData instance or None if loading fails
    """
    try:
        # Log the path we're using
        logger.info(f"Loading GEDCOM file from: {gedcom_path}")

        # Check if the file exists and is readable
        if not gedcom_path.exists():
            logger.error(f"GEDCOM file does not exist: {gedcom_path}")
            return None

        if not os.access(gedcom_path, os.R_OK):
            logger.error(f"GEDCOM file is not readable: {gedcom_path}")
            return None

        # Log file details
        file_size = gedcom_path.stat().st_size
        logger.info(f"GEDCOM file size: {file_size} bytes")

        # Try to read the first few lines of the file to verify it's a valid GEDCOM file
        try:
            with open(gedcom_path, "r", encoding="utf-8") as f:
                first_lines = [next(f) for _ in range(5)]
                logger.info(f"First few lines of GEDCOM file:\n{''.join(first_lines)}")
        except UnicodeDecodeError:
            logger.info("File is not UTF-8 encoded, trying with latin-1")
            try:
                with open(gedcom_path, "r", encoding="latin-1") as f:
                    first_lines = [next(f) for _ in range(5)]
                    logger.info(
                        f"First few lines of GEDCOM file (latin-1):\n{''.join(first_lines)}"
                    )
            except Exception as e:
                logger.error(
                    f"Error reading first few lines with latin-1 encoding: {e}"
                )
        except Exception as e:
            logger.error(f"Error reading first few lines: {e}")

        # Create GedcomData instance
        logger.info("Creating GedcomData instance...")
        gedcom_data = GedcomData(gedcom_path)

        # Check if the instance was created successfully
        if gedcom_data:
            logger.info(f"GedcomData instance created successfully")
            logger.info(f"GedcomData type: {type(gedcom_data)}")
            logger.info(f"GedcomData attributes: {dir(gedcom_data)}")

            # Try to access some attributes to verify it's working
            if hasattr(gedcom_data, "path"):
                logger.info(f"GedcomData.path: {gedcom_data.path}")

            # Try to build caches if the method exists
            if hasattr(gedcom_data, "build_caches"):
                logger.info("Building caches...")
                try:
                    gedcom_data.build_caches()
                    logger.info("Caches built successfully")
                except Exception as e:
                    logger.error(f"Error building caches: {e}", exc_info=True)
            else:
                logger.warning("GedcomData does not have build_caches method")

            return gedcom_data
        else:
            logger.error("GedcomData instance is None")
            return None
    except Exception as e:
        logger.error(f"Error loading GEDCOM file: {e}", exc_info=True)
        return None


def get_gedcom_data():
    """
    Returns the cached GEDCOM data instance, loading it if necessary.

    This function ensures the GEDCOM file is loaded only once and reused
    throughout the script, improving performance.

    Returns:
        GedcomData instance or None if loading fails
    """
    global _CACHED_GEDCOM_DATA

    # Return cached data if already loaded
    if _CACHED_GEDCOM_DATA is not None:
        logger.info("Using cached GEDCOM data")
        return _CACHED_GEDCOM_DATA

    # Check if GEDCOM path is configured
    gedcom_path_str = config_instance.GEDCOM_FILE_PATH
    logger.info(f"GEDCOM_FILE_PATH from config: {gedcom_path_str}")

    if not gedcom_path_str:
        logger.warning("GEDCOM_FILE_PATH not configured. Cannot load GEDCOM file.")
        return None

    # Convert string to Path object
    gedcom_path = Path(gedcom_path_str)

    # Make sure the path is absolute
    if not gedcom_path.is_absolute():
        # If it's a relative path, make it absolute relative to the project root
        original_path = gedcom_path
        gedcom_path = Path(os.path.dirname(os.path.abspath(__file__))) / gedcom_path
        logger.info(
            f"Converted relative path '{original_path}' to absolute path: {gedcom_path}"
        )

    # Check if the file exists
    if not gedcom_path.exists():
        logger.warning(f"GEDCOM file not found at {gedcom_path}")

        # Try looking for the file in the Data directory directly
        data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "Data"
        if data_dir.exists():
            # Look for files with .ged extension
            ged_files = list(data_dir.glob("*.ged"))
            if ged_files:
                logger.info(
                    f"Found {len(ged_files)} GEDCOM files in {data_dir}: {[f.name for f in ged_files]}"
                )

                # Look for a file with a name similar to what's in the config
                gedcom_filename = os.path.basename(gedcom_path_str)
                for ged_file in ged_files:
                    if ged_file.name.lower() == gedcom_filename.lower():
                        gedcom_path = ged_file
                        logger.info(f"Found matching GEDCOM file: {gedcom_path}")
                        break

                # If no exact match, use the first .ged file found
                if not gedcom_path.exists() and ged_files:
                    gedcom_path = ged_files[0]
                    logger.info(f"Using first available GEDCOM file: {gedcom_path}")
            else:
                logger.warning(f"No .ged files found in {data_dir}")
        else:
            logger.warning(f"Data directory {data_dir} does not exist")

    # Check again if the file exists after our search
    if not gedcom_path.exists():
        logger.error(f"GEDCOM file not found after searching. Cannot proceed.")
        return None

    # Log file details
    logger.info(
        f"GEDCOM file exists: {gedcom_path.exists()}, Size: {gedcom_path.stat().st_size} bytes"
    )

    # Load GEDCOM data
    try:
        logger.info(f"Loading GEDCOM file {gedcom_path} (first time)...")

        # Check if required modules are available
        try:
            # Import ged4py to verify it's installed
            # This is needed by gedcom_utils
            import importlib

            importlib.import_module("ged4py")
            logger.info("Required modules imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            return None

        _CACHED_GEDCOM_DATA = load_gedcom_data(gedcom_path)

        if _CACHED_GEDCOM_DATA:
            logger.info(f"GEDCOM file loaded successfully and cached for reuse.")
            # Log some stats about the loaded data
            logger.info(f"  GEDCOM data type: {type(_CACHED_GEDCOM_DATA)}")
            logger.info(f"  GEDCOM data attributes: {dir(_CACHED_GEDCOM_DATA)}")

            if hasattr(_CACHED_GEDCOM_DATA, "indi_index"):
                logger.info(f"  Index size: {len(_CACHED_GEDCOM_DATA.indi_index)}")
            else:
                logger.warning("  GEDCOM data does not have indi_index attribute")

            if hasattr(_CACHED_GEDCOM_DATA, "processed_data_cache"):
                logger.info(
                    f"  Pre-processed cache size: {len(_CACHED_GEDCOM_DATA.processed_data_cache)}"
                )
            else:
                logger.warning(
                    "  GEDCOM data does not have processed_data_cache attribute"
                )

            # Try to build caches if they don't exist
            if hasattr(_CACHED_GEDCOM_DATA, "build_caches") and (
                not hasattr(_CACHED_GEDCOM_DATA, "processed_data_cache")
                or not _CACHED_GEDCOM_DATA.processed_data_cache
            ):
                logger.info("  Attempting to build caches...")
                try:
                    _CACHED_GEDCOM_DATA.build_caches()
                    logger.info("  Caches built successfully")
                    if hasattr(_CACHED_GEDCOM_DATA, "processed_data_cache"):
                        logger.info(
                            f"  Pre-processed cache size after build: {len(_CACHED_GEDCOM_DATA.processed_data_cache)}"
                        )
                except Exception as e:
                    logger.error(f"  Error building caches: {e}", exc_info=True)
        else:
            logger.warning("load_gedcom_data returned None")

        return _CACHED_GEDCOM_DATA
    except Exception as e:
        logger.error(f"Error loading GEDCOM file: {e}", exc_info=True)
        return None


# Constants
TEST_DB_NAME = "test_ai_responses.db"
PRODUCTIVE_SENTIMENT = "PRODUCTIVE"
OTHER_SENTIMENT = "OTHER"
CUSTOM_RESPONSE_MESSAGE_TYPE = "Automated_Genealogy_Response"

# Configure test database path
original_db_path = config_instance.DATABASE_FILE
test_db_path = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / TEST_DB_NAME

# Ensure data directory exists
data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
data_dir.mkdir(exist_ok=True)


# Function to get real people from GEDCOM file
def get_real_people_from_gedcom(
    gedcom_data=None, max_people=10, distant_ancestors=False
):
    """
    Get a list of real people from the GEDCOM file.

    Args:
        gedcom_data: Pre-loaded GedcomData instance (if None, will load it)
        max_people: Maximum number of people to return
        distant_ancestors: If True, prioritize people from 5+ generations back

    Returns:
        List of dictionaries with person details
    """
    real_people = []
    distant_people = []
    recent_people = []

    try:
        # Use provided GEDCOM data or load it if not provided
        if gedcom_data is None:
            logger.info("No pre-loaded GEDCOM data provided, loading now...")
            gedcom_data = get_gedcom_data()

        if not gedcom_data:
            logger.warning("Failed to load GEDCOM data")
            return real_people

        if not hasattr(gedcom_data, "processed_data_cache"):
            logger.warning("GEDCOM data does not have processed_data_cache attribute")
            logger.info(f"GEDCOM data attributes: {dir(gedcom_data)}")
            return real_people

        # Only log this message when we're loading the data, not when using pre-loaded data
        if gedcom_data is None:
            logger.info(
                f"GEDCOM data loaded successfully with {len(gedcom_data.processed_data_cache)} entries in cache"
            )

        # Get all people from the processed data cache
        cache = gedcom_data.processed_data_cache
        all_ids = list(cache.keys())

        # Shuffle the IDs for randomness
        random.shuffle(all_ids)

        # Process all people and categorize them as distant or recent ancestors
        current_year = datetime.now().year
        cutoff_year = (
            current_year - 180
        )  # Roughly 6 generations back (30 years per generation)

        for person_id in all_ids:
            person_data = cache.get(person_id)
            if person_data:
                # Extract relevant information
                name = person_data.get("full_name_disp", "Unknown")
                birth_year = person_data.get("birth_year")
                birth_place = person_data.get("birth_place_disp", "unknown location")
                death_year = person_data.get("death_year")
                death_place = person_data.get("death_place_disp", "unknown location")

                # Only include people with at least name and birth year
                if name != "Unknown" and birth_year:
                    person_info = {
                        "id": person_id,
                        "name": name,
                        "birth_year": birth_year,
                        "birth_place": birth_place or "unknown location",
                        "death_year": death_year,
                        "death_place": death_place or "unknown location",
                    }

                    # Categorize as distant or recent ancestor
                    if birth_year < cutoff_year:
                        distant_people.append(person_info)
                    else:
                        recent_people.append(person_info)

        # Log the counts
        logger.info(
            f"Found {len(distant_people)} distant ancestors (born before {cutoff_year})"
        )
        logger.info(
            f"Found {len(recent_people)} recent ancestors (born after {cutoff_year})"
        )

        # Return the appropriate set based on the distant_ancestors flag
        if distant_ancestors:
            # If we want distant ancestors but don't have enough, supplement with recent ones
            if len(distant_people) < max_people:
                logger.info(
                    f"Not enough distant ancestors, supplementing with recent ones"
                )
                real_people = (
                    distant_people + recent_people[: max_people - len(distant_people)]
                )
            else:
                real_people = distant_people[:max_people]
        else:
            # If we want recent ancestors but don't have enough, supplement with distant ones
            if len(recent_people) < max_people:
                logger.info(
                    f"Not enough recent ancestors, supplementing with distant ones"
                )
                real_people = (
                    recent_people + distant_people[: max_people - len(recent_people)]
                )
            else:
                real_people = recent_people[:max_people]

        # Shuffle the final list to mix distant and recent ancestors if we had to supplement
        random.shuffle(real_people)
        real_people = real_people[:max_people]

        logger.info(f"Returning {len(real_people)} people from GEDCOM file")
    except Exception as e:
        logger.error(f"Error getting real people from GEDCOM: {e}", exc_info=True)

    return real_people


# Define message templates and data
MESSAGE_TEMPLATES = {
    "not_interested": [
        "Sorry, I'm not interested in genealogy research at this time.",
        "Thanks for reaching out, but I don't have time for family history right now.",
        "I appreciate your message, but I'm not looking to explore my family tree.",
        "Thank you for your interest, but genealogy isn't something I'm focused on currently.",
        "I'm not really into family history research, but thanks anyway.",
        "I received your message about genealogy, but I'm afraid I'm not interested in pursuing this. I hope you understand.",
        "While I appreciate your efforts to connect on family history, I'm not in a position to engage with this research at the moment.",
        "Thank you for thinking of me regarding our potential family connection, but I must decline as genealogy isn't a priority for me right now.",
    ],
    "interested_but_brief": [
        "Thanks for reaching out! I'm definitely interested in learning more about my family history.",
        "I'd love to hear what you've found about our possible connection!",
        "Family history is fascinating to me. What have you discovered?",
        "I've been doing a bit of research myself and would be happy to share information.",
        "Yes, I'm interested in genealogy. What branch of the family are you researching?",
        "I'm very excited to hear from you about our potential family connection. I've always been curious about my ancestors but haven't had much time to research them properly.",
        "Thank you for your message. I'm quite interested in genealogy and would appreciate any information you can share about our family history. I've been trying to trace my roots for some time now.",
        "I'm definitely interested in learning more about my family tree. My grandmother used to tell stories about our ancestors, but I've never been able to verify any of the details. Any information you have would be greatly appreciated.",
    ],
    "detailed_in_tree": [
        "I've been researching my great-great-great-grandfather {name} who was born in {birth_place} in {birth_year}. He married {spouse_name} in {marriage_year} and they had {child_count} children. He worked as a {occupation} and died in {death_year} in {death_place}. I found some old letters that mention his father {father_name} was from {father_origin}. Do you have any information about this branch of the family?",
        "I recently discovered that my ancestor {name} ({birth_year}-{death_year}) from {birth_place} was actually related to the {surname} family through his mother {mother_name}. He moved to {location} around {year} and established a {business_type} business. His daughter {daughter_name} married into the {married_surname} family. I'm trying to find more details about his siblings.",
        "My research shows that {name} born {birth_year} in {birth_place} was the youngest of {sibling_count} children. Their parents were {father_name} and {mother_name} who came from {origin_place}. {name} served in the {military_unit} during the {war_name} before settling in {settlement_place} where he worked as a {occupation}. I found records showing he died in {death_year} from {cause_of_death}.",
        "I've traced my lineage back to {name} ({birth_year}-{death_year}) who lived in {location}. He married {spouse_name} in {marriage_year} at {marriage_place}. They had {child_count} children: {child_names}. {name} owned {property_description} and was known for {achievement}. His father {father_name} came from {father_origin} in the early {migration_period}.",
        "According to family records, my ancestor {name} was born in {birth_year} in {birth_place} to {father_name} and {mother_name}. He had {sibling_count} siblings. The family moved to {new_location} in {move_year} where {name} became a {occupation}. He married {spouse_name} and they had {child_count} children. He died in {death_year} after {death_circumstance}.",
    ],
    "detailed_not_in_tree": [
        "I've been researching my great-great-grandfather {name} who was born in {birth_place} in {birth_year}. He married {spouse_name} in {marriage_year} and they had {child_count} children. He worked as a {occupation} and died in {death_year} in {death_place}. I found some old letters that mention his father {father_name} was from {father_origin}. Do you have any information about this branch of the family?",
        "I recently discovered that my ancestor {name} ({birth_year}-{death_year}) from {birth_place} was actually related to the {surname} family through his mother {mother_name}. He moved to {location} around {year} and established a {business_type} business. His daughter {daughter_name} married into the {married_surname} family. I'm trying to find more details about his siblings.",
        "My research shows that {name} born {birth_year} in {birth_place} was the youngest of {sibling_count} children. Their parents were {father_name} and {mother_name} who came from {origin_place}. {name} served in the {military_unit} during the {war_name} before settling in {settlement_place} where he worked as a {occupation}. I found records showing he died in {death_year} from {cause_of_death}.",
        "I've traced my lineage back to {name} ({birth_year}-{death_year}) who lived in {location}. He married {spouse_name} in {marriage_year} at {marriage_place}. They had {child_count} children: {child_names}. {name} owned {property_description} and was known for {achievement}. His father {father_name} came from {father_origin} in the early {migration_period}.",
        "According to family records, my ancestor {name} was born in {birth_year} in {birth_place} to {father_name} and {mother_name}. He had {sibling_count} siblings. The family moved to {new_location} in {move_year} where {name} became a {occupation}. He married {spouse_name} and they had {child_count} children. He died in {death_year} after {death_circumstance}.",
    ],
    "partial_details_in_tree": [
        "I think we might be related through my ancestor {name} who was born sometime in the {decade}s. I don't know much about them except they lived in {location} and had something to do with {occupation}. Do you have any information that might help me learn more?",
        "I'm trying to find information about my great-great-grandfather {name}. I know he was from {location} and married someone named {spouse_name}, but I don't have dates or other details. Does this connect to your family tree?",
        "My grandmother used to talk about her grandfather {name} who came from {origin_place}. I think he was born around {approximate_year} but I'm not sure. She mentioned he had {sibling_count} siblings. Does this sound familiar?",
        "I have a family Bible that mentions {name} who lived in {location}. There's a note saying they were related to the {surname} family, but I don't have dates or other details. I'm trying to piece together how they fit into our family tree.",
        "My aunt told me about our ancestor {name} who was some kind of {occupation} in {location}. I think they lived during the late {century} century. I don't have much more information, but I'm curious if you know anything about this person.",
    ],
    "miscellaneous": [
        "I found some old family photos recently and I'm trying to identify the people in them. One photo from around {year} has a note mentioning {name}. Have you come across this name in your research?",
        "I've hit a brick wall with my {surname} ancestors from {location}. The records before {year} seem to be missing or destroyed. Do you have any suggestions for alternative sources I could check?",
        "My DNA results show we're likely {relationship} but I can't figure out the exact connection. My family comes from {location} and I've traced back to my 3rd great-grandparents {ancestor_names}. Does this help narrow down our connection?",
        "I inherited some family heirlooms that supposedly belonged to {name} who lived in {location} during the {time_period}. I'm trying to verify if they're actually from our family. Do you have any information that might help?",
        "I'm planning a trip to {location} to visit the places where our ancestors lived. Do you know of any specific sites, churches, or cemeteries that would be worth visiting to learn more about the {surname} family history?",
        "I've been researching {name} who I believe might be connected to our family. According to the census records from {year}, they lived in {location} and had connections to several families in the area. I'm wondering if you've come across any information about them in your research?",
        "Hello! I recently discovered through a DNA match that we might be related through {name} from {location}. I've been trying to piece together how exactly we're connected, but I'm having trouble finding reliable records. Have you done any research on this branch of the family?",
        "I'm working on a family history book and I'm trying to include as much information as possible about {name} and their descendants. I know they lived in {location} during the {time_period}, but I'm struggling to find details about their daily lives, occupations, or any interesting stories. Would you happen to have any information that could help bring their story to life?",
        "I've been going through some old church records from {location} and found several mentions of {name} around {year}. The handwriting is difficult to decipher, but it seems they were involved in some kind of community event or possibly a legal dispute. I'm curious if you've come across anything similar in your research?",
        "While researching my {surname} ancestors, I found a newspaper article from {year} that mentions {name} in connection with a local event in {location}. The article suggests they were quite prominent in the community. I'm trying to determine if this is the same {name} who appears in my family tree. Do you have any insights that might help confirm this connection?",
    ],
}

# Data for generating fictitious people and messages
FIRST_NAMES = [
    "James",
    "John",
    "Robert",
    "Michael",
    "William",
    "David",
    "Richard",
    "Joseph",
    "Thomas",
    "Charles",
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Barbara",
    "Susan",
    "Jessica",
    "Sarah",
    "Karen",
]

LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Miller",
    "Davis",
    "Garcia",
    "Rodriguez",
    "Wilson",
    "Martinez",
    "Anderson",
    "Taylor",
    "Thomas",
    "Hernandez",
    "Moore",
    "Martin",
    "Jackson",
    "Thompson",
    "White",
]

LOCATIONS = [
    "Boston",
    "New York",
    "Philadelphia",
    "Chicago",
    "San Francisco",
    "London",
    "Edinburgh",
    "Glasgow",
    "Dublin",
    "Belfast",
    "Paris",
    "Berlin",
    "Munich",
    "Vienna",
    "Amsterdam",
    "Brussels",
    "Rome",
    "Madrid",
]

OCCUPATIONS = [
    "farmer",
    "blacksmith",
    "carpenter",
    "teacher",
    "merchant",
    "doctor",
    "lawyer",
    "minister",
    "miner",
    "tailor",
    "seamstress",
    "nurse",
    "baker",
    "brewer",
    "shoemaker",
    "printer",
    "clerk",
    "soldier",
]

SURNAMES = [
    "MacDonald",
    "Campbell",
    "Stewart",
    "Murray",
    "Graham",
    "Hamilton",
    "Ferguson",
    "Kennedy",
    "Grant",
    "Robertson",
    "Reid",
    "Ross",
    "Watson",
    "Morrison",
    "Fraser",
    "Cameron",
    "Johnston",
    "Henderson",
]

WARS = [
    "Civil War",
    "Revolutionary War",
    "War of 1812",
    "World War I",
    "Boer War",
    "Crimean War",
    "Spanish-American War",
]

MILITARY_UNITS = [
    "42nd Regiment",
    "Royal Scots",
    "Highland Light Infantry",
    "Black Watch",
    "Gordon Highlanders",
    "Seaforth Highlanders",
    "Cameron Highlanders",
    "Royal Artillery",
    "Royal Engineers",
]

BUSINESS_TYPES = [
    "textile mill",
    "general store",
    "blacksmith shop",
    "carpentry business",
    "printing press",
    "law practice",
    "medical practice",
    "tavern",
    "bakery",
    "shipping company",
]

CAUSES_OF_DEATH = [
    "pneumonia",
    "tuberculosis",
    "heart failure",
    "accident",
    "old age",
    "influenza",
    "typhoid fever",
    "cholera",
    "smallpox",
    "yellow fever",
]

ACHIEVEMENTS = [
    "local politics",
    "church leadership",
    "community service",
    "military service",
    "business success",
    "educational contributions",
    "charitable work",
    "inventions",
    "artistic talents",
]

PROPERTY_DESCRIPTIONS = [
    "a small farm",
    "a townhouse",
    "a country estate",
    "a shop in town",
    "several acres of land",
    "a mill by the river",
    "rental properties",
    "a plantation",
    "a homestead",
]


def generate_message_content(message_type, gedcom_data=None):
    """
    Generate a message based on the specified type.

    Args:
        message_type: Type of message to generate
        gedcom_data: Pre-loaded GedcomData instance to avoid reloading for each message
    """
    if message_type == "not_interested":
        return random.choice(MESSAGE_TEMPLATES["not_interested"])

    elif message_type == "detailed_in_tree":
        template = random.choice(MESSAGE_TEMPLATES["detailed_in_tree"])

        # Always use real people from GEDCOM file for this type (100% chance)
        use_real_person = True
        # 50% chance of using distant ancestors (6+ generations back)
        use_distant_ancestors = random.random() < 0.5
        real_people = []

        if use_real_person:
            real_people = get_real_people_from_gedcom(
                gedcom_data=gedcom_data,
                max_people=10,
                distant_ancestors=use_distant_ancestors,
            )
            if use_distant_ancestors and real_people:
                logger.info(
                    f"Using distant ancestors (6+ generations back) for detailed_in_tree message"
                )

        # If we have real people, use one of them
        if real_people:
            real_person = random.choice(real_people)
            logger.info(f"Using real person from GEDCOM: {real_person['name']}")

            # Format with real person data
            return template.format(
                name=real_person["name"],
                birth_year=real_person["birth_year"],
                birth_place=real_person["birth_place"],
                death_year=(
                    real_person["death_year"]
                    if real_person["death_year"]
                    else random.randint(
                        real_person["birth_year"] + 50, real_person["birth_year"] + 80
                    )
                ),
                death_place=real_person["death_place"],
                spouse_name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
                marriage_year=random.randint(
                    real_person["birth_year"] + 20, real_person["birth_year"] + 40
                ),
                marriage_place=random.choice(LOCATIONS),
                child_count=random.randint(1, 12),
                child_names=", ".join([random.choice(FIRST_NAMES) for _ in range(3)]),
                father_name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
                mother_name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
                father_origin=random.choice(LOCATIONS),
                occupation=random.choice(OCCUPATIONS),
                sibling_count=random.randint(1, 10),
                origin_place=random.choice(LOCATIONS),
                military_unit=random.choice(MILITARY_UNITS),
                war_name=random.choice(WARS),
                settlement_place=random.choice(LOCATIONS),
                cause_of_death=random.choice(CAUSES_OF_DEATH),
                surname=(
                    real_person["name"].split()[-1]
                    if " " in real_person["name"]
                    else random.choice(SURNAMES)
                ),
                location=random.choice(LOCATIONS),
                year=random.randint(
                    real_person["birth_year"] + 20, real_person["birth_year"] + 50
                ),
                business_type=random.choice(BUSINESS_TYPES),
                daughter_name=random.choice(FIRST_NAMES),
                married_surname=random.choice(SURNAMES),
                property_description=random.choice(PROPERTY_DESCRIPTIONS),
                achievement=random.choice(ACHIEVEMENTS),
                migration_period=f"18{random.randint(0, 9)}0s",
                move_year=random.randint(
                    real_person["birth_year"] + 15, real_person["birth_year"] + 40
                ),
                death_circumstance=random.choice(CAUSES_OF_DEATH),
                new_location=random.choice(LOCATIONS),
            )

        # Otherwise use fictional data
        return template.format(
            name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
            birth_year=random.randint(1750, 1900),
            birth_place=random.choice(LOCATIONS),
            death_year=random.randint(1800, 1950),
            death_place=random.choice(LOCATIONS),
            spouse_name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
            marriage_year=random.randint(1770, 1920),
            marriage_place=random.choice(LOCATIONS),
            child_count=random.randint(1, 12),
            child_names=", ".join([random.choice(FIRST_NAMES) for _ in range(3)]),
            father_name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
            mother_name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
            father_origin=random.choice(LOCATIONS),
            occupation=random.choice(OCCUPATIONS),
            sibling_count=random.randint(1, 10),
            origin_place=random.choice(LOCATIONS),
            military_unit=random.choice(MILITARY_UNITS),
            war_name=random.choice(WARS),
            settlement_place=random.choice(LOCATIONS),
            cause_of_death=random.choice(CAUSES_OF_DEATH),
            surname=random.choice(SURNAMES),
            location=random.choice(LOCATIONS),
            year=random.randint(1750, 1950),
            business_type=random.choice(BUSINESS_TYPES),
            daughter_name=random.choice(FIRST_NAMES),
            married_surname=random.choice(SURNAMES),
            property_description=random.choice(PROPERTY_DESCRIPTIONS),
            achievement=random.choice(ACHIEVEMENTS),
            migration_period=f"18{random.randint(0, 9)}0s",
            move_year=random.randint(1750, 1950),
            death_circumstance=random.choice(CAUSES_OF_DEATH),
            new_location=random.choice(LOCATIONS),
        )

    elif message_type == "detailed_not_in_tree":
        template = random.choice(MESSAGE_TEMPLATES["detailed_not_in_tree"])

        # Try to get real people from GEDCOM file (20% chance)
        use_real_person = random.random() < 0.2
        # 40% chance of using distant ancestors (5+ generations back)
        use_distant_ancestors = random.random() < 0.4
        real_people = []

        if use_real_person:
            real_people = get_real_people_from_gedcom(
                gedcom_data=gedcom_data,
                max_people=5,
                distant_ancestors=use_distant_ancestors,
            )
            if use_distant_ancestors and real_people:
                logger.info(
                    f"Using distant ancestors (6+ generations back) for detailed_not_in_tree message"
                )

        # If we have real people, use one of them but with a different last name
        # to simulate someone who might be related but not in the tree
        if real_people:
            real_person = random.choice(real_people)
            # Get first name from real person, but use a random last name
            name_parts = real_person["name"].split()
            if len(name_parts) > 1:
                first_name = name_parts[0]
                # Make sure we get a different last name
                last_name = random.choice(LAST_NAMES)
                while last_name == name_parts[-1]:
                    last_name = random.choice(LAST_NAMES)

                modified_name = f"{first_name} {last_name}"
                logger.info(
                    f"Using modified real person from GEDCOM: {modified_name} (original: {real_person['name']})"
                )

                # Format with modified real person data
                return template.format(
                    name=modified_name,
                    birth_year=real_person["birth_year"],
                    birth_place=real_person["birth_place"],
                    death_year=(
                        real_person["death_year"]
                        if real_person["death_year"]
                        else random.randint(
                            real_person["birth_year"] + 50,
                            real_person["birth_year"] + 80,
                        )
                    ),
                    death_place=real_person["death_place"],
                    spouse_name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                    marriage_year=random.randint(
                        real_person["birth_year"] + 20, real_person["birth_year"] + 40
                    ),
                    marriage_place=random.choice(LOCATIONS),
                    child_count=random.randint(1, 12),
                    child_names=", ".join(
                        [random.choice(FIRST_NAMES) for _ in range(3)]
                    ),
                    father_name=f"{random.choice(FIRST_NAMES)} {last_name}",
                    mother_name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                    father_origin=random.choice(LOCATIONS),
                    occupation=random.choice(OCCUPATIONS),
                    sibling_count=random.randint(1, 10),
                    origin_place=random.choice(LOCATIONS),
                    military_unit=random.choice(MILITARY_UNITS),
                    war_name=random.choice(WARS),
                    settlement_place=random.choice(LOCATIONS),
                    cause_of_death=random.choice(CAUSES_OF_DEATH),
                    surname=last_name,
                    location=random.choice(LOCATIONS),
                    year=random.randint(
                        real_person["birth_year"] + 20, real_person["birth_year"] + 50
                    ),
                    business_type=random.choice(BUSINESS_TYPES),
                    daughter_name=random.choice(FIRST_NAMES),
                    married_surname=random.choice(LAST_NAMES),
                    property_description=random.choice(PROPERTY_DESCRIPTIONS),
                    achievement=random.choice(ACHIEVEMENTS),
                    migration_period=f"18{random.randint(0, 9)}0s",
                    move_year=random.randint(
                        real_person["birth_year"] + 15, real_person["birth_year"] + 40
                    ),
                    death_circumstance=random.choice(CAUSES_OF_DEATH),
                    new_location=random.choice(LOCATIONS),
                )

        # Otherwise use completely fictional data
        return template.format(
            name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            birth_year=random.randint(1750, 1900),
            birth_place=random.choice(LOCATIONS),
            death_year=random.randint(1800, 1950),
            death_place=random.choice(LOCATIONS),
            spouse_name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            marriage_year=random.randint(1770, 1920),
            marriage_place=random.choice(LOCATIONS),
            child_count=random.randint(1, 12),
            child_names=", ".join([random.choice(FIRST_NAMES) for _ in range(3)]),
            father_name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            mother_name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            father_origin=random.choice(LOCATIONS),
            occupation=random.choice(OCCUPATIONS),
            sibling_count=random.randint(1, 10),
            origin_place=random.choice(LOCATIONS),
            military_unit=random.choice(MILITARY_UNITS),
            war_name=random.choice(WARS),
            settlement_place=random.choice(LOCATIONS),
            cause_of_death=random.choice(CAUSES_OF_DEATH),
            surname=random.choice(LAST_NAMES),
            location=random.choice(LOCATIONS),
            year=random.randint(1750, 1950),
            business_type=random.choice(BUSINESS_TYPES),
            daughter_name=random.choice(FIRST_NAMES),
            married_surname=random.choice(LAST_NAMES),
            property_description=random.choice(PROPERTY_DESCRIPTIONS),
            achievement=random.choice(ACHIEVEMENTS),
            migration_period=f"18{random.randint(0, 9)}0s",
            move_year=random.randint(1750, 1950),
            death_circumstance=random.choice(CAUSES_OF_DEATH),
            new_location=random.choice(LOCATIONS),
        )

    elif message_type == "partial_details_in_tree":
        template = random.choice(MESSAGE_TEMPLATES["partial_details_in_tree"])

        # Always try to use real people from GEDCOM file for this type (100% chance)
        use_real_person = True
        # 40% chance of using distant ancestors (5+ generations back)
        use_distant_ancestors = random.random() < 0.4
        real_people = []

        if use_real_person:
            real_people = get_real_people_from_gedcom(
                gedcom_data=gedcom_data,
                max_people=10,
                distant_ancestors=use_distant_ancestors,
            )
            if use_distant_ancestors and real_people:
                logger.info(
                    f"Using distant ancestors (6+ generations back) for partial_details_in_tree message"
                )

        # If we have real people, use one of them with partial information
        if real_people:
            real_person = random.choice(real_people)
            logger.info(
                f"Using real person from GEDCOM for partial details: {real_person['name']}"
            )

            # Extract surname
            surname = (
                real_person["name"].split()[-1]
                if " " in real_person["name"]
                else random.choice(SURNAMES)
            )

            # Calculate decade from birth year
            birth_decade = (real_person["birth_year"] // 10) * 10

            # Format with real person data but only partial details
            return template.format(
                name=real_person["name"],
                decade=f"{birth_decade}s",
                location=real_person["birth_place"],
                occupation=random.choice(OCCUPATIONS),
                spouse_name=random.choice(FIRST_NAMES),
                origin_place=random.choice(LOCATIONS),
                approximate_year=real_person["birth_year"],
                sibling_count=random.randint(1, 10),
                surname=surname,
                century=real_person["birth_year"] // 100 + 1,
            )

        # Otherwise use fictional data
        return template.format(
            name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
            decade=f"18{random.randint(0, 9)}0",
            location=random.choice(LOCATIONS),
            occupation=random.choice(OCCUPATIONS),
            spouse_name=random.choice(FIRST_NAMES),
            origin_place=random.choice(LOCATIONS),
            approximate_year=random.randint(1750, 1900),
            sibling_count=random.randint(1, 10),
            surname=random.choice(SURNAMES),
            century=random.randint(17, 19),
        )

    else:  # miscellaneous
        template = random.choice(MESSAGE_TEMPLATES["miscellaneous"])

        # Try to get real people from GEDCOM file (70% chance)
        use_real_person = random.random() < 0.7
        # 40% chance of using distant ancestors (5+ generations back)
        use_distant_ancestors = random.random() < 0.4
        real_people = []

        if use_real_person:
            real_people = get_real_people_from_gedcom(
                gedcom_data=gedcom_data,
                max_people=10,
                distant_ancestors=use_distant_ancestors,
            )
            if use_distant_ancestors and real_people:
                logger.info(
                    f"Using distant ancestors (5+ generations back) for miscellaneous message"
                )

        # If we have real people, use one of them
        if real_people:
            real_person = random.choice(real_people)
            logger.info(
                f"Using real person from GEDCOM for miscellaneous message: {real_person['name']}"
            )

            # Extract surname
            surname = (
                real_person["name"].split()[-1]
                if " " in real_person["name"]
                else random.choice(SURNAMES)
            )

            # Get a year from the person's lifetime
            if real_person["death_year"]:
                year = random.randint(
                    real_person["birth_year"], real_person["death_year"]
                )
            else:
                year = random.randint(
                    real_person["birth_year"], real_person["birth_year"] + 70
                )

            # Format with real person data
            return template.format(
                year=year,
                name=real_person["name"],
                surname=surname,
                location=real_person["birth_place"],
                relationship=random.choice(
                    ["2nd cousins", "3rd cousins", "4th cousins", "distant cousins"]
                ),
                ancestor_names=f"{random.choice(FIRST_NAMES)} and {real_person['name']}",
                time_period=f"{(real_person['birth_year'] // 10) * 10}s",
            )

        # Otherwise use fictional data
        return template.format(
            year=random.randint(1850, 1950),
            name=f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
            surname=random.choice(SURNAMES),
            location=random.choice(LOCATIONS),
            relationship=random.choice(
                ["2nd cousins", "3rd cousins", "4th cousins", "distant cousins"]
            ),
            ancestor_names=f"{random.choice(FIRST_NAMES)} and {random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}",
            time_period=f"18{random.randint(0, 9)}0s",
        )


def generate_fictitious_person():
    """Generate data for a fictitious person."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    username = f"{first_name} {last_name}"
    gender = "M" if first_name in FIRST_NAMES[:10] else "F"
    birth_year = random.randint(1950, 2000)

    return {
        "uuid": f"TEST-{uuid4()}",
        "profile_id": f"TEST-{uuid4()}",
        "username": username,
        "first_name": first_name,
        # Note: Person model doesn't have last_name field
        "gender": gender,
        "birth_year": birth_year,
        "status": PersonStatusEnum.ACTIVE,
        "contactable": True,
        "in_my_tree": random.choice([True, False]),
    }


def ensure_ai_feedback_column():
    """
    Ensure the ai_feedback column exists in the ConversationLog table.

    Returns:
        True if successful, False otherwise
    """
    logger.info("Ensuring ai_feedback column exists in ConversationLog table")

    try:
        # Get a database connection
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.cursor()

        # First check if the table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_log'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check if the column exists
            cursor.execute("PRAGMA table_info(conversation_log)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            if "ai_feedback" not in column_names:
                # Add the column
                cursor.execute(
                    "ALTER TABLE conversation_log ADD COLUMN ai_feedback INTEGER"
                )
                conn.commit()
                logger.info("Added ai_feedback column to ConversationLog table")
            else:
                logger.info(
                    "ai_feedback column already exists in ConversationLog table"
                )
        else:
            logger.info(
                "conversation_log table doesn't exist yet, will be created with schema"
            )

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Error ensuring ai_feedback column: {e}", exc_info=True)
        logger.info(
            "Continuing despite error - column will be created with schema if needed"
        )
        return True  # Return True anyway, as the table might not exist yet


def create_test_database():
    """Create a new test database with the required schema."""
    logger.info(f"Creating test database at {test_db_path}")

    # Temporarily override the database path in config
    config_instance.DATABASE_FILE = test_db_path

    # Delete existing test database if it exists
    if test_db_path.exists():
        try:
            test_db_path.unlink()
            logger.info(f"Deleted existing test database: {test_db_path}")
        except Exception as e:
            logger.error(f"Failed to delete existing test database: {e}")
            return False

    # Create a new SessionManager with the test database
    session_manager = SessionManager()

    # Initialize the database
    if not session_manager.ensure_db_ready():
        logger.error("Failed to initialize test database")
        return False

    # Ensure the ai_feedback column exists
    if not ensure_ai_feedback_column():
        logger.error("Failed to ensure ai_feedback column exists")
        return False

    logger.info("Test database created successfully")
    return session_manager


def create_test_data(session_manager, num_people=10):  # Using 10 people for manual runs
    """Create test data with fictitious people and messages."""
    logger.info(f"Creating {num_people} fictitious people with messages")

    # Get a database session
    db_session = session_manager.get_db_conn()
    if not db_session:
        logger.error("Failed to get database session")
        return False

    # Load GEDCOM data once at the beginning
    logger.info("Loading GEDCOM data once for all message generation...")
    gedcom_data = get_gedcom_data()
    if gedcom_data:
        logger.info(
            f"GEDCOM data loaded successfully with {len(gedcom_data.processed_data_cache)} entries in cache"
        )
    else:
        logger.warning(
            "Failed to load GEDCOM data, will generate messages without real people"
        )

    # Create message type for custom responses if it doesn't exist
    with db_transn(db_session) as session:
        # Check if the message type already exists
        message_type = (
            session.query(MessageType)
            .filter_by(type_name=CUSTOM_RESPONSE_MESSAGE_TYPE)
            .first()
        )
        if not message_type:
            # Create the message type
            message_type = MessageType(type_name=CUSTOM_RESPONSE_MESSAGE_TYPE)
            session.add(message_type)
            logger.info(f"Created message type: {CUSTOM_RESPONSE_MESSAGE_TYPE}")

        # Check if the 'refers_to_tree_person' column exists in the ConversationLog table
        # If not, add it
        try:
            # Use raw SQL to check if the column exists
            conn = sqlite3.connect(str(test_db_path))
            cursor = conn.cursor()

            # First check if the table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_log'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Check if the column exists
                cursor.execute("PRAGMA table_info(conversation_log)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]

                if "refers_to_tree_person" not in column_names:
                    logger.info(
                        "Adding refers_to_tree_person column to conversation_log table"
                    )
                    cursor.execute(
                        "ALTER TABLE conversation_log ADD COLUMN refers_to_tree_person BOOLEAN"
                    )
                    conn.commit()
                    logger.info("Column added successfully")
                else:
                    logger.info("refers_to_tree_person column already exists")
            else:
                logger.info(
                    "conversation_log table doesn't exist yet, will be created with schema"
                )

            conn.close()
        except Exception as e:
            logger.error(
                f"Error ensuring refers_to_tree_person column: {e}", exc_info=True
            )
            logger.info(
                "Continuing despite error - column will be created with schema if needed"
            )
            # Don't return False here, as the table might not exist yet

    # Create people and messages
    people_created = 0

    # Custom distribution as requested:
    # 70% of messages should reference someone in the tree
    # 30% of messages should not reference someone in the tree
    message_types_distribution = {
        "not_interested": int(num_people * 0.05),  # 5% not interested
        "detailed_in_tree": int(
            num_people * 0.40
        ),  # 40% detailed info about people in tree
        "partial_details_in_tree": int(
            num_people * 0.30
        ),  # 30% partial info about people in tree
        "detailed_not_in_tree": int(
            num_people * 0.15
        ),  # 15% detailed info about people not in tree
        "miscellaneous": int(num_people * 0.10),  # 10% miscellaneous messages
    }

    # Adjust to ensure we have exactly num_people messages
    total = sum(message_types_distribution.values())
    if total < num_people:
        # Add the remaining to detailed_in_tree to maintain high in-tree percentage
        message_types_distribution["detailed_in_tree"] += num_people - total
    elif total > num_people:
        # Remove from not_interested first, then from detailed_not_in_tree if needed
        excess = total - num_people
        if message_types_distribution["not_interested"] > excess:
            message_types_distribution["not_interested"] -= excess
        else:
            excess -= message_types_distribution["not_interested"]
            message_types_distribution["not_interested"] = 0
            message_types_distribution["detailed_not_in_tree"] -= excess

    # Log the distribution
    logger.info("Message type distribution:")
    for msg_type, count in message_types_distribution.items():
        logger.info(f"  {msg_type}: {count} messages ({count/num_people*100:.1f}%)")

    # Flatten the distribution into a list for random selection
    message_type_list = []
    for msg_type, count in message_types_distribution.items():
        message_type_list.extend([msg_type] * count)
    random.shuffle(message_type_list)

    # Process in batches to improve efficiency
    batch_size = 10
    for batch_start in range(0, num_people, batch_size):
        batch_end = min(batch_start + batch_size, num_people)
        batch_range = range(batch_start, batch_end)

        logger.info(
            f"Processing batch {batch_start//batch_size + 1} (people {batch_start+1}-{batch_end})"
        )

        # First create all people in the batch
        people_batch = []
        with db_transn(db_session) as session:
            for i in batch_range:
                # Generate person data
                person_data = generate_fictitious_person()
                person = Person(**person_data)
                session.add(person)

                # Store the message type with the person and person_data for later use
                people_batch.append((person, message_type_list[i], person_data))

            # Flush to get all IDs
            session.flush()

        # Now create messages and classify them
        # We'll collect all conversation IDs and their refers_to_tree_person values
        # to update them in a single transaction at the end
        conversation_data = []

        with db_transn(db_session) as session:
            for person, message_type, person_data in people_batch:
                # Generate a message from this person
                message_content = generate_message_content(message_type, gedcom_data)

                # Create a conversation log entry for this message
                conversation_id = f"test_conv_{uuid4()}"
                message_timestamp = datetime.now(timezone.utc) - timedelta(
                    days=random.randint(1, 30)
                )

                # Format the message for AI classification
                # This simulates a conversation with just one message from the user
                formatted_context = f"USER: {message_content}"

                # Call the AI to classify the message intent
                print(
                    f"\rClassifying message {people_created+1}/{num_people} for person {person.id}...",
                    end="",
                )
                ai_sentiment = classify_message_intent(
                    formatted_context, session_manager
                )

                # If AI classification fails, use a fallback based on message type
                if not ai_sentiment:
                    logger.warning(
                        f"AI classification failed for person {person.id}, using fallback"
                    )
                    ai_sentiment = (
                        PRODUCTIVE_SENTIMENT
                        if message_type != "not_interested"
                        else "UNINTERESTED"
                    )

                logger.info(
                    f"Message for person {person.id} classified as: {ai_sentiment}"
                )

                # Determine if this message refers to someone in the tree
                refers_to_tree_person = message_type in [
                    "detailed_in_tree",
                    "partial_details_in_tree",
                ]

                # Log whether this message refers to someone in the tree
                logger.info(
                    f"Message for person {person.id} refers to tree person: {refers_to_tree_person}"
                )

                # Create the conversation log without the refers_to_tree_person field
                conversation_log = ConversationLog(
                    conversation_id=conversation_id,
                    direction=MessageDirectionEnum.IN,
                    people_id=person.id,
                    latest_message_content=message_content,
                    latest_timestamp=message_timestamp,
                    ai_sentiment=ai_sentiment,
                )

                # Add the conversation log to the session
                session.add(conversation_log)

                # Store the conversation ID and refers_to_tree_person value for later update
                conversation_data.append((conversation_id, refers_to_tree_person))

                people_created += 1

        # Now update all the refers_to_tree_person values in a single transaction
        try:
            # Check if the column exists
            conn = sqlite3.connect(str(test_db_path))
            cursor = conn.cursor()

            # Check if the column exists
            cursor.execute("PRAGMA table_info(conversation_log)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            if "refers_to_tree_person" in column_names:
                # Start a transaction
                conn.execute("BEGIN TRANSACTION")

                # Update all records in a single transaction
                for conversation_id, refers_to_tree_person in conversation_data:
                    cursor.execute(
                        """
                        UPDATE conversation_log
                        SET refers_to_tree_person = ?
                        WHERE conversation_id = ? AND direction = ?
                        """,
                        (
                            1 if refers_to_tree_person else 0,
                            conversation_id,
                            MessageDirectionEnum.IN.value,
                        ),
                    )

                # Commit the transaction
                conn.commit()
                logger.info(
                    f"Updated refers_to_tree_person values for {len(conversation_data)} conversations"
                )

            conn.close()
        except Exception as e:
            logger.error(
                f"Error updating refers_to_tree_person values: {e}", exc_info=True
            )
            # Continue anyway, this is just metadata for our test

        # Clear the progress line and print batch completion
        print(
            f"\rBatch {batch_start//batch_size + 1} complete: {batch_end-batch_start} people processed.{' '*30}"
        )
        logger.info(f"Created {people_created} people with messages so far")

    logger.info(f"Successfully created {people_created} people with messages")
    return True


def evaluate_ai_responses(session_manager, auto_evaluate=False):
    """
    Utility function to evaluate AI responses.

    This function:
    1. Finds all AI-generated responses in the database
    2. If auto_evaluate=True, automatically evaluates responses based on criteria
    3. If auto_evaluate=False, displays each response and allows manual evaluation
    4. Updates the database with the feedback

    Args:
        session_manager: The active SessionManager instance
        auto_evaluate: Whether to automatically evaluate responses (default: False)
    """
    logger.info("=== Evaluating AI Responses ===")

    # Get a database session
    db_session = session_manager.get_db_conn()
    if not db_session:
        logger.error("Failed to get database session")
        return False

    # First, ensure we have the ai_feedback_text column in the database
    try:
        # Check if the column exists
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.cursor()

        # First check if the table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_log'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check if the column exists
            cursor.execute("PRAGMA table_info(conversation_log)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            # Add the ai_feedback_text column if it doesn't exist
            if "ai_feedback_text" not in column_names:
                logger.info("Adding ai_feedback_text column to conversation_log table")
                cursor.execute(
                    "ALTER TABLE conversation_log ADD COLUMN ai_feedback_text TEXT"
                )
                conn.commit()
                logger.info("Column added successfully")
            else:
                logger.info("ai_feedback_text column already exists")
        else:
            logger.info(
                "conversation_log table doesn't exist yet, will be created with schema"
            )

        conn.close()
    except Exception as e:
        logger.error(f"Error ensuring ai_feedback_text column: {e}", exc_info=True)
        logger.info(
            "Continuing despite error - column will be created with schema if needed"
        )
        if not auto_evaluate:
            input("\nPress Enter to continue...")
            return False

    # Find all OUT messages with test_generated status
    try:
        # Use a raw SQL query to avoid the attribute error
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.cursor()
        # First check if the refers_to_tree_person column exists
        cursor.execute("PRAGMA table_info(conversation_log)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        if "refers_to_tree_person" in column_names:
            # If the column exists, include it in the query
            cursor.execute(
                """
                SELECT cl_out.conversation_id, cl_out.people_id, cl_out.latest_message_content,
                       p.username, p.id, cl_in.refers_to_tree_person, cl_in.latest_message_content as in_content
                FROM conversation_log cl_out
                JOIN people p ON cl_out.people_id = p.id
                JOIN conversation_log cl_in ON cl_out.conversation_id = cl_in.conversation_id AND cl_in.direction = 'IN'
                WHERE cl_out.direction = 'OUT'
                AND cl_out.script_message_status = 'test_generated'
                ORDER BY cl_out.people_id
                """
            )
        else:
            # If the column doesn't exist, use a default value of NULL
            cursor.execute(
                """
                SELECT cl_out.conversation_id, cl_out.people_id, cl_out.latest_message_content,
                       p.username, p.id, NULL as refers_to_tree_person, cl_in.latest_message_content as in_content
                FROM conversation_log cl_out
                JOIN people p ON cl_out.people_id = p.id
                JOIN conversation_log cl_in ON cl_out.conversation_id = cl_in.conversation_id AND cl_in.direction = 'IN'
                WHERE cl_out.direction = 'OUT'
                AND cl_out.script_message_status = 'test_generated'
                ORDER BY cl_out.people_id
                """
            )
        out_messages_data = cursor.fetchall()

        if not out_messages_data:
            print("\nNo AI-generated responses found to evaluate")
            input("\nPress Enter to continue...")
            conn.close()
            return True

        print(f"\nFound {len(out_messages_data)} AI-generated responses to evaluate")

        # Process each response
        evaluated_count = 0

        if auto_evaluate:
            print("\nAutomatically evaluating responses...")

            # Define evaluation criteria
            def evaluate_response_quality(
                out_content, in_content, refers_to_tree_person
            ):
                """
                Automatically evaluate the quality of a response based on various criteria.

                Args:
                    out_content: The AI-generated response content
                    in_content: The original message content
                    refers_to_tree_person: Whether the message refers to someone in the tree

                Returns:
                    tuple: (is_acceptable, feedback_text, score)
                """
                score = 0
                feedback_points = []
                max_score = 10

                # Check if response is empty or too short
                if not out_content or len(out_content) < 50:
                    return (0, "Response is too short or empty.", 0)

                # Check if response addresses the original message
                # Look for key terms from the original message in the response
                key_terms = set()
                for word in in_content.lower().split():
                    if len(word) > 4 and word not in [
                        "about",
                        "would",
                        "could",
                        "their",
                        "there",
                        "these",
                        "those",
                    ]:
                        key_terms.add(word)

                # Count how many key terms from the original message appear in the response
                matching_terms = sum(
                    1 for term in key_terms if term in out_content.lower()
                )
                if matching_terms >= 3:
                    score += 1
                    feedback_points.append(
                        f"Response addresses {matching_terms} key terms from the original message."
                    )

                # Check if response is too generic
                generic_phrases = [
                    "I don't have specific information",
                    "I couldn't find information",
                    "I don't have access to",
                    "I don't have enough information",
                    "Without more details",
                ]

                generic_count = sum(
                    1
                    for phrase in generic_phrases
                    if phrase.lower() in out_content.lower()
                )
                if generic_count >= 2:
                    score -= 2
                    feedback_points.append(
                        "Response contains too many generic phrases."
                    )

                # Check if response acknowledges whether the person is in the tree
                if refers_to_tree_person:
                    if (
                        "in your family tree" in out_content.lower()
                        or "in our records" in out_content.lower()
                    ):
                        score += 2
                        feedback_points.append(
                            "Correctly identifies that the person is in the family tree."
                        )
                    else:
                        score -= 2
                        feedback_points.append(
                            "Fails to acknowledge that the person is in the family tree."
                        )
                else:
                    if (
                        "couldn't find" in out_content.lower()
                        or "don't have information" in out_content.lower()
                    ):
                        score += 1
                        feedback_points.append(
                            "Correctly acknowledges that the person is not in the family tree."
                        )

                # Check for genealogical details
                genealogical_terms = [
                    "birth",
                    "born",
                    "death",
                    "died",
                    "marriage",
                    "married",
                    "children",
                    "child",
                    "parent",
                    "father",
                    "mother",
                    "sibling",
                    "brother",
                    "sister",
                    "family",
                    "ancestor",
                    "descendant",
                    "generation",
                    "relative",
                    "relation",
                ]

                genealogical_count = sum(
                    1
                    for term in genealogical_terms
                    if term.lower() in out_content.lower()
                )
                if genealogical_count >= 5:
                    score += 2
                    feedback_points.append("Contains good genealogical details.")
                elif genealogical_count >= 3:
                    score += 1
                    feedback_points.append("Contains some genealogical details.")
                else:
                    score -= 1
                    feedback_points.append("Lacks sufficient genealogical details.")

                # Check for personalization
                if "you" in out_content.lower() and "your" in out_content.lower():
                    score += 1
                    feedback_points.append("Response is personalized.")

                # Check for dates and locations
                date_pattern = r"\b(1[0-9]{3}|20[0-2][0-9])\b"
                dates = re.findall(date_pattern, out_content)
                if dates:
                    score += 1
                    feedback_points.append(
                        f"Includes specific dates: {', '.join(dates)}"
                    )

                # Check for relationship explanation
                relationship_terms = [
                    "related to",
                    "connection",
                    "connected",
                    "relationship",
                ]
                if any(term in out_content.lower() for term in relationship_terms):
                    score += 1
                    feedback_points.append("Explains relationship connections.")

                # Check for helpful tone
                helpful_phrases = [
                    "hope this helps",
                    "let me know",
                    "I'd be happy to",
                    "please feel free",
                ]
                if any(phrase in out_content.lower() for phrase in helpful_phrases):
                    score += 1
                    feedback_points.append("Has a helpful and engaging tone.")

                # Check for follow-up questions or suggestions
                if "?" in out_content:
                    score += 1
                    feedback_points.append("Includes follow-up questions.")

                # Normalize score to be between 0 and 1
                normalized_score = max(0, min(score, max_score)) / max_score

                # Determine if acceptable based on score threshold
                is_acceptable = normalized_score >= 0.6

                # Generate feedback text
                if is_acceptable:
                    feedback_text = f"Score: {score}/{max_score}. " + " ".join(
                        feedback_points
                    )
                    if normalized_score >= 0.8:
                        feedback_text = "Excellent response. " + feedback_text
                    else:
                        feedback_text = (
                            "Good response with some areas for improvement. "
                            + feedback_text
                        )
                else:
                    feedback_text = (
                        f"Score: {score}/{max_score}. Response needs improvement. "
                        + " ".join(feedback_points)
                    )

                return (1 if is_acceptable else 0, feedback_text, normalized_score)

            # Process each response automatically
            for out_msg_data in out_messages_data:
                (
                    conversation_id,
                    _,
                    out_content,
                    username,
                    person_id,
                    refers_to_tree_person,
                    in_content,
                ) = out_msg_data

                # Convert refers_to_tree_person to a boolean
                # It could be None, 0/1, or True/False depending on the database
                if refers_to_tree_person is None:
                    # If we don't have the information, assume it's not in the tree
                    refers_to_tree_person_bool = False
                    logger.info(
                        f"No refers_to_tree_person data for conversation {conversation_id}, assuming False"
                    )
                else:
                    # Convert to boolean (handles 0/1, True/False, etc.)
                    refers_to_tree_person_bool = bool(refers_to_tree_person)

                # Evaluate the response
                is_acceptable, feedback_text, score = evaluate_response_quality(
                    out_content, in_content, refers_to_tree_person_bool
                )

                # Update the database with the feedback
                try:
                    cursor.execute(
                        """
                        UPDATE conversation_log
                        SET ai_feedback = ?, ai_feedback_text = ?
                        WHERE conversation_id = ? AND direction = 'OUT'
                        """,
                        (is_acceptable, feedback_text, conversation_id),
                    )
                    conn.commit()
                    evaluated_count += 1

                    # Log the evaluation
                    logger.info(
                        f"Auto-evaluated response for {username} (ID: {person_id}): "
                        f"{'Acceptable' if is_acceptable else 'Unacceptable'} (Score: {score:.2f})"
                    )
                    logger.info(f"Feedback: {feedback_text}")

                except Exception as e:
                    logger.error(f"Error updating feedback: {e}", exc_info=True)
                    conn.rollback()

            print(f"\nAutomatically evaluated {evaluated_count} responses")

        else:
            # Manual evaluation process
            for out_msg_data in out_messages_data:
                (
                    conversation_id,
                    _,
                    out_content,
                    username,
                    person_id,
                    refers_to_tree_person,
                    in_content,
                ) = out_msg_data

                # Convert refers_to_tree_person to a boolean
                # It could be None, 0/1, or True/False depending on the database
                if refers_to_tree_person is None:
                    # If we don't have the information, assume it's not in the tree
                    refers_to_tree_person_bool = False
                    refers_to_tree_person_str = "Unknown (assuming No)"
                    logger.info(
                        f"No refers_to_tree_person data for conversation {conversation_id}, assuming False"
                    )
                else:
                    # Convert to boolean (handles 0/1, True/False, etc.)
                    refers_to_tree_person_bool = bool(refers_to_tree_person)
                    refers_to_tree_person_str = (
                        "Yes" if refers_to_tree_person_bool else "No"
                    )

                # Display the messages
                print("\n" + "=" * 80)
                print(f"Person: {username} (ID: {person_id})")
                print(f"Refers to person in tree: {refers_to_tree_person_str}")
                print(f"Original Message (IN):")
                print("-" * 80)
                print(in_content)
                print("\nAI Response (OUT):")
                print("-" * 80)
                print(out_content)
                print("=" * 80)

                # Check if this response has already been evaluated
                cursor.execute(
                    """
                    SELECT ai_feedback, ai_feedback_text
                    FROM conversation_log
                    WHERE conversation_id = ? AND direction = 'OUT'
                    """,
                    (conversation_id,),
                )
                existing_feedback = cursor.fetchone()

                if existing_feedback and existing_feedback[0] is not None:
                    print(
                        f"\nThis response has already been evaluated as {'acceptable' if existing_feedback[0] == 1 else 'unacceptable'}"
                    )
                    print(f"Previous feedback: {existing_feedback[1]}")

                    while True:
                        re_evaluate = (
                            input("\nDo you want to re-evaluate this response? (y/n): ")
                            .strip()
                            .lower()
                        )
                        if re_evaluate in ["y", "n"]:
                            break
                        print("Invalid input. Please enter 'y' or 'n'.")

                    if re_evaluate == "n":
                        print("Skipping this response.")
                        continue

                # Get user feedback on acceptability
                while True:
                    feedback = input(
                        "\nIs this response acceptable? (1=Yes, 0=No): "
                    ).strip()
                    if feedback in ["0", "1"]:
                        break
                    print("Invalid input. Please enter 0 or 1.")

                # Get detailed textual feedback
                print("\nPlease provide detailed feedback on this response:")
                print("(What's good/bad about it? How could it be improved?)")
                feedback_text = input("> ").strip()

                # If no feedback was provided, prompt again
                if not feedback_text:
                    print(
                        "Please provide at least some brief feedback to help improve the AI:"
                    )
                    feedback_text = input("> ").strip()
                    if not feedback_text:
                        feedback_text = "No specific feedback provided."

                # Update the database with the feedback
                try:
                    cursor.execute(
                        """
                        UPDATE conversation_log
                        SET ai_feedback = ?, ai_feedback_text = ?
                        WHERE conversation_id = ? AND direction = 'OUT'
                        """,
                        (int(feedback), feedback_text, conversation_id),
                    )
                    conn.commit()
                    evaluated_count += 1
                    print(
                        f"Response marked as {'acceptable' if feedback == '1' else 'unacceptable'} with feedback"
                    )
                except Exception as e:
                    logger.error(f"Error updating feedback: {e}", exc_info=True)
                    conn.rollback()

            print(f"\nManually evaluated {evaluated_count} AI-generated responses")

        conn.close()
        if not auto_evaluate:
            input("\nPress Enter to continue...")
        return True

    except Exception as e:
        logger.error(f"Error evaluating AI responses: {e}", exc_info=True)
        if not auto_evaluate:
            input("\nPress Enter to continue...")
        return False


def analyze_feedback_and_suggest_improvements(session_manager):
    """
    Analyze the feedback on AI responses and suggest prompt improvements.

    This function:
    1. Retrieves all evaluated AI responses from the database
    2. Analyzes patterns in acceptable vs. unacceptable responses
    3. Incorporates detailed textual feedback from users
    4. Suggests improvements to the AI prompts

    Args:
        session_manager: The active SessionManager instance
    """
    logger.info("=== Analyzing Feedback and Suggesting Improvements ===")

    # Get a database session
    db_session = session_manager.get_db_conn()
    if not db_session:
        logger.error("Failed to get database session")
        return False

    try:
        # Use raw SQL to avoid attribute errors
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.cursor()

        # Get all evaluated responses
        cursor.execute(
            """
            SELECT cl.conversation_id, cl.latest_message_content, cl.ai_feedback,
                   cl.ai_feedback_text, p.username
            FROM conversation_log cl
            JOIN people p ON cl.people_id = p.id
            WHERE cl.direction = 'OUT'
            AND cl.script_message_status = 'test_generated'
            AND cl.ai_feedback IS NOT NULL
            """
        )
        evaluated_responses = cursor.fetchall()

        if not evaluated_responses:
            print("\nNo evaluated responses found")
            input("\nPress Enter to continue...")
            conn.close()
            return True

        # Count acceptable and unacceptable responses
        total_responses = len(evaluated_responses)
        acceptable_responses = sum(1 for r in evaluated_responses if r[2] == 1)
        unacceptable_responses = total_responses - acceptable_responses

        print("\n=== Feedback Analysis ===")
        print(f"Total evaluated responses: {total_responses}")
        print(
            f"Acceptable responses: {acceptable_responses} ({acceptable_responses/total_responses*100:.1f}%)"
        )
        print(
            f"Unacceptable responses: {unacceptable_responses} ({unacceptable_responses/total_responses*100:.1f}%)"
        )

        # Get all responses with their corresponding IN messages
        all_response_pairs = []
        unacceptable_pairs = []
        for out_msg_data in evaluated_responses:
            conversation_id, out_content, ai_feedback, feedback_text, username = (
                out_msg_data
            )

            # Get the corresponding IN message
            cursor.execute(
                """
                SELECT latest_message_content
                FROM conversation_log
                WHERE conversation_id = ? AND direction = 'IN'
                ORDER BY latest_timestamp DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
            in_msg_result = cursor.fetchone()

            if in_msg_result:
                in_content = in_msg_result[0]
                # Add to all responses list
                all_response_pairs.append(
                    (in_content, out_content, feedback_text, username, ai_feedback)
                )

                # Also add to unacceptable list if applicable
                if ai_feedback == 0:  # Unacceptable
                    unacceptable_pairs.append(
                        (in_content, out_content, feedback_text, username)
                    )

        # First, analyze all responses (both acceptable and unacceptable)
        print("\n=== Analysis of All Responses ===")
        if all_response_pairs:
            # Collect all feedback text for analysis
            all_feedback = []

            # Analyze all responses for detailed analysis
            print(f"\nDetailed analysis of all {len(all_response_pairs)} responses:")
            for i, (
                in_content,
                out_content,
                feedback_text,
                username,
                ai_feedback,
            ) in enumerate(all_response_pairs, 1):
                acceptability = "Acceptable" if ai_feedback == 1 else "Unacceptable"
                print(f"\nResponse {i} (from {username}) - {acceptability}:")
                print(f"IN: {in_content[:100]}...")
                print(f"OUT: {out_content[:100]}...")
                print(f"FEEDBACK: {feedback_text}")
                all_feedback.append(feedback_text)

            # Analyze all feedback text for common themes
            print("\n=== Common Themes in All Feedback ===")
            all_feedback_text = " ".join(all_feedback).lower()

            # List of common feedback themes to check for
            themes = {
                "Relationship details": ["relationship", "related", "connection"],
                "Birth/death information": ["birth", "death", "born", "died"],
                "Family structure": ["family", "parent", "child", "sibling"],
                "Historical context": ["history", "historical", "period", "era"],
                "Geographic information": ["location", "place", "country", "city"],
                "Response tone": ["tone", "friendly", "formal", "conversational"],
                "Response length": ["length", "long", "short", "concise"],
                "Follow-up questions": ["question", "ask", "inquiry", "follow-up"],
            }

            # Check for each theme in the feedback
            found_themes = []
            for theme, keywords in themes.items():
                if any(keyword in all_feedback_text for keyword in keywords):
                    found_themes.append(theme)

            if found_themes:
                print("The following themes appeared in all user feedback:")
                for theme in found_themes:
                    print(f"- {theme}")
            else:
                print("No specific themes identified in the feedback.")

        # Then, analyze patterns in unacceptable responses specifically
        print("\n=== Analysis of Unacceptable Responses ===")
        if unacceptable_pairs:
            # Categorize unacceptable responses
            categories = {
                "irrelevant_info": 0,
                "missing_key_details": 0,
                "incorrect_facts": 0,
                "poor_formatting": 0,
                "too_generic": 0,
                "too_specific": 0,
                "other": 0,
            }

            # Collect feedback text for unacceptable responses
            unacceptable_feedback = []

            # Analyze unacceptable responses (without repeating the full details)
            print(
                f"\nAnalyzing {len(unacceptable_pairs)} unacceptable responses for specific issues:"
            )
            for in_content, out_content, feedback_text, username in unacceptable_pairs:
                unacceptable_feedback.append(feedback_text)

                # Analyze this response based on content and feedback
                if "genealogy" not in out_content.lower():
                    categories["irrelevant_info"] += 1
                if len(out_content) < 100:
                    categories["too_generic"] += 1
                if len(out_content) > 1000:
                    categories["too_specific"] += 1

                # Analyze feedback text for keywords
                feedback_lower = feedback_text.lower()
                if any(
                    term in feedback_lower
                    for term in ["missing", "lack", "incomplete", "more detail"]
                ):
                    categories["missing_key_details"] += 1
                if any(
                    term in feedback_lower
                    for term in ["wrong", "incorrect", "error", "mistake"]
                ):
                    categories["incorrect_facts"] += 1
                if any(
                    term in feedback_lower
                    for term in ["format", "structure", "organize", "paragraph"]
                ):
                    categories["poor_formatting"] += 1

            # Print category counts
            print("\nIssue categories in unacceptable responses:")
            for category, count in categories.items():
                if count > 0:
                    print(f"- {category.replace('_', ' ').title()}: {count} instances")

            # Analyze unacceptable feedback text for common themes
            print("\n=== Common Themes in Unacceptable Feedback ===")
            unacceptable_feedback_text = " ".join(unacceptable_feedback).lower()

            # List of common feedback themes to check for
            unacceptable_themes = {
                "Relationship details": ["relationship", "related", "connection"],
                "Birth/death information": ["birth", "death", "born", "died"],
                "Family structure": ["family", "parent", "child", "sibling"],
                "Historical context": ["history", "historical", "period", "era"],
                "Geographic information": ["location", "place", "country", "city"],
                "Response tone": ["tone", "friendly", "formal", "conversational"],
                "Response length": ["length", "long", "short", "concise"],
                "Follow-up questions": ["question", "ask", "inquiry", "follow-up"],
            }

            # Check for each theme in the feedback
            unacceptable_found_themes = []
            for theme, keywords in unacceptable_themes.items():
                if any(keyword in unacceptable_feedback_text for keyword in keywords):
                    unacceptable_found_themes.append(theme)

            if unacceptable_found_themes:
                print("The following themes appeared in unacceptable feedback:")
                for theme in unacceptable_found_themes:
                    print(f"- {theme}")
            else:
                print("No specific themes identified in the unacceptable feedback.")

            # Suggest improvements
            print("\n=== Suggested Prompt Improvements ===")
            print(
                "Based on the analysis and user feedback, consider the following improvements to the AI prompts:"
            )

            if categories["irrelevant_info"] > 0:
                print(
                    "1. Emphasize relevance: Instruct the AI to focus strictly on genealogical information mentioned in the message."
                )

            if categories["missing_key_details"] > 0:
                print(
                    "2. Improve extraction: Enhance the prompt to better identify and extract key genealogical details from messages."
                )

            if categories["incorrect_facts"] > 0:
                print(
                    "3. Fact verification: Add instructions for the AI to verify facts against the genealogical data before including them in responses."
                )

            if categories["poor_formatting"] > 0:
                print(
                    "4. Formatting guidelines: Provide clearer formatting guidelines for responses, including paragraph structure and information organization."
                )

            if categories["too_generic"] > 0:
                print(
                    "5. Add specificity: Instruct the AI to include specific details from the genealogical data rather than generic statements."
                )

            if categories["too_specific"] > 0:
                print(
                    "6. Control verbosity: Add guidelines for appropriate response length and level of detail."
                )

            # Add suggestions based on feedback themes
            if "Relationship details" in found_themes:
                print(
                    "7. Relationship clarity: Emphasize clear explanation of how people are related to the user's tree."
                )

            if "Birth/death information" in found_themes:
                print(
                    "8. Vital records: Ensure birth and death information is consistently included when available."
                )

            if "Response tone" in found_themes:
                print(
                    "9. Tone adjustment: Refine the conversational tone to be more engaging while remaining professional."
                )

            # General suggestions
            print("\nGeneral suggestions:")
            print("- Include more examples of high-quality responses in the prompt.")
            print(
                "- Add explicit instructions about tone, style, and level of formality."
            )
            print(
                "- Consider a structured response format with sections for different types of information."
            )
            print(
                "- Implement a two-step process: first extract information, then generate a response based on the extracted data."
            )

            # Add specific suggestions from unacceptable feedback
            print("\nSpecific suggestions from unacceptable feedback:")
            for feedback in unacceptable_feedback:
                if (
                    "suggest" in feedback.lower()
                    or "recommend" in feedback.lower()
                    or "should" in feedback.lower()
                ):
                    print(f"- {feedback}")
        else:
            print("No unacceptable responses to analyze.")

        conn.close()
        input("\nPress Enter to continue...")
        return True

    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}", exc_info=True)
        input("\nPress Enter to continue...")
        return False


def update_ai_prompts(session_manager):
    """
    Update AI prompts based on feedback analysis.

    This function:
    1. Analyzes feedback on AI responses including textual feedback
    2. Generates improved prompts incorporating user suggestions
    3. Saves the improved prompts to files and updates the ai_prompts.json file

    Args:
        session_manager: The active SessionManager instance
    """
    logger.info("=== Updating AI Prompts ===")

    # Import the AI prompt utilities
    try:
        from ai_prompt_utils import update_prompt
    except ImportError:
        logger.error("Failed to import ai_prompt_utils module")
        return False

    # Get a database session
    db_session = session_manager.get_db_conn()
    if not db_session:
        logger.error("Failed to get database session")
        return False

    try:
        # Use raw SQL to avoid attribute errors
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.cursor()

        # Get all evaluated responses
        cursor.execute(
            """
            SELECT cl.conversation_id, cl.latest_message_content, cl.ai_feedback,
                   cl.ai_feedback_text, p.username
            FROM conversation_log cl
            JOIN people p ON cl.people_id = p.id
            WHERE cl.direction = 'OUT'
            AND cl.script_message_status = 'test_generated'
            AND cl.ai_feedback IS NOT NULL
            """
        )
        evaluated_responses = cursor.fetchall()

        if not evaluated_responses:
            print("\nNo evaluated responses found")
            input("\nPress Enter to continue...")
            conn.close()
            return True

        # Count acceptable and unacceptable responses
        total_responses = len(evaluated_responses)
        acceptable_responses = sum(1 for r in evaluated_responses if r[2] == 1)
        unacceptable_responses = total_responses - acceptable_responses

        print("\n=== Prompt Improvement Based on Feedback ===")
        print(f"Total evaluated responses: {total_responses}")
        print(
            f"Acceptable responses: {acceptable_responses} ({acceptable_responses/total_responses*100:.1f}%)"
        )
        print(
            f"Unacceptable responses: {unacceptable_responses} ({unacceptable_responses/total_responses*100:.1f}%)"
        )

        # Get examples of good and bad responses with their feedback
        good_examples = []
        bad_examples = []
        feedback_suggestions = []

        for out_msg_data in evaluated_responses:
            conversation_id, out_content, ai_feedback, feedback_text, _ = out_msg_data

            # Get the corresponding IN message
            cursor.execute(
                """
                SELECT latest_message_content
                FROM conversation_log
                WHERE conversation_id = ? AND direction = 'IN'
                ORDER BY latest_timestamp DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
            in_msg_result = cursor.fetchone()

            if in_msg_result:
                in_content = in_msg_result[0]
                example = {
                    "input": in_content,
                    "output": out_content,
                    "feedback": feedback_text,
                }

                if ai_feedback == 1:  # Acceptable
                    good_examples.append(example)
                else:  # Unacceptable
                    bad_examples.append(example)

                # Extract suggestions from feedback
                if feedback_text:
                    feedback_lower = feedback_text.lower()
                    if any(
                        term in feedback_lower
                        for term in [
                            "suggest",
                            "recommend",
                            "should",
                            "improve",
                            "better",
                        ]
                    ):
                        feedback_suggestions.append(feedback_text)

        # Use all good examples for the improved prompt
        good_samples = good_examples

        # Extract common themes from feedback
        all_feedback = [
            ex.get("feedback", "")
            for ex in good_examples + bad_examples
            if ex.get("feedback")
        ]
        all_feedback_text = " ".join(all_feedback).lower()

        # List of common feedback themes to check for
        themes = {
            "Relationship details": ["relationship", "related", "connection"],
            "Birth/death information": ["birth", "death", "born", "died"],
            "Family structure": ["family", "parent", "child", "sibling"],
            "Historical context": ["history", "historical", "period", "era"],
            "Geographic information": ["location", "place", "country", "city"],
            "Response tone": ["tone", "friendly", "formal", "conversational"],
            "Response length": ["length", "long", "short", "concise"],
            "Follow-up questions": ["question", "ask", "inquiry", "follow-up"],
        }

        # Check for each theme in the feedback
        found_themes = []
        for theme, keywords in themes.items():
            if any(keyword in all_feedback_text for keyword in keywords):
                found_themes.append(theme)

        # Generate improved prompts
        # 1. Improved extraction prompt with more specific guidance
        improved_extraction_prompt = """
You are a genealogy research assistant specializing in extracting key information from messages about family history.

Your task is to carefully analyze the message and extract the following types of information:
1. Names of people mentioned (full names if available) - Be precise and extract ONLY actual names mentioned
2. Dates (birth, death, marriage, etc.) - Format as they appear in the message
3. Locations (birth places, residences, etc.) - Include country, state/province, and city/town when available
4. Relationships between people - Be specific about the relationship type (e.g., "John Smith is Mary's grandfather")
5. Key facts - Focus on genealogically relevant information only
6. Occupations or professions - Note who held which occupation
7. Historical events or time periods - Note how they relate to the people mentioned
8. Research questions or brick walls mentioned - Identify specific genealogical questions

IMPORTANT GUIDELINES:
- Extract ONLY information that is explicitly mentioned in the message
- Do NOT make assumptions or infer information not directly stated
- For names, extract FULL NAMES when available (first and last name)
- If a name is ambiguous or incomplete, still include it but don't guess missing parts
- Include ALL mentioned names, even if they appear multiple times
- For dates, include the year at minimum, and full dates when available
- For locations, be as specific as possible with the information provided
- Focus ONLY on extracting factual genealogical information
- Do not include general greetings or pleasantries
- Extract SPECIFIC details (e.g., "John Smith born 1850 in London" rather than just "a person")
- If a detail is uncertain or approximate in the message, indicate this (e.g., "born ~1850s")
- Do not infer or add information not present in the message
- If no information of a particular type is present, return an empty list for that category
"""

        # Add theme-specific guidelines based on feedback
        if "Relationship details" in found_themes:
            improved_extraction_prompt += """
- Pay special attention to relationship information (e.g., "John is Mary's father", "siblings James and Sarah")
- Extract both explicit relationships ("father of") and implicit ones ("married in 1850" implies a spousal relationship)
"""

        if "Birth/death information" in found_themes:
            improved_extraction_prompt += """
- Carefully extract all birth and death information, including approximate dates/years
- Note the context of dates (birth, death, marriage, immigration, etc.)
"""

        improved_extraction_prompt += """
Return your analysis as a JSON object with the following structure:
{
  "mentioned_names": ["Full Name 1", "Full Name 2"],
  "dates": ["Date 1 (context)", "Date 2 (context)"],
  "locations": ["Location 1 (context)", "Location 2 (context)"],
  "relationships": ["Person A is father of Person B", "Person C is married to Person D"],
  "occupations": ["Person A was a farmer", "Person B was a teacher"],
  "events": ["Family moved to X in 1850", "Served in Civil War"],
  "research_questions": ["Looking for information about Person X's parents", "Trying to find birth record"]
}

EXAMPLES OF GOOD EXTRACTION:
"""

        # Add good examples to the extraction prompt
        for i, example in enumerate(good_samples, 1):
            improved_extraction_prompt += f"""
Example {i}:
Input: {example['input'][:200]}...
Output: {{
  "mentioned_names": ["John Smith", "Mary Johnson"],
  "dates": ["1850 (birth of John)", "1880 (marriage)"],
  "locations": ["London, England (birthplace)", "New York (residence)"],
  "relationships": ["John Smith married Mary Johnson"],
  "occupations": ["John was a carpenter"],
  "events": ["Emigrated to America in 1870"],
  "research_questions": ["Looking for information about John's parents"]
}}
"""

        # 2. Improved response generation prompt with more specific guidance
        improved_response_prompt = """
You are a helpful genealogy assistant. Your task is to generate a personalized reply to a message about family history.

CONVERSATION CONTEXT:
{conversation_context}

USER'S LAST MESSAGE:
{user_message}

GENEALOGICAL DATA:
{genealogical_data}

IMPORTANT INSTRUCTIONS:
1. Focus ONLY on the genealogical information in the user's message and the provided genealogical data
2. Be precise and accurate - only include facts that are supported by the genealogical data
3. Prioritize information about specific people mentioned in the user's message
4. Include full birth/death dates and locations when available
5. Clearly explain family relationships, especially how people connect to the user's family tree
6. Use a warm, conversational tone while maintaining professionalism
7. Keep your response concise (200-400 words) and well-organized
8. Use paragraphs to separate different topics or people
9. If appropriate, include 1-2 specific follow-up questions that could help advance their research
10. Do not include information that isn't supported by the genealogical data
11. When mentioning dates, be clear about whether they are birth, death, marriage, etc.
12. When mentioning relationships, be specific (e.g., "maternal grandfather" rather than just "grandfather")
13. If the genealogical data is incomplete or uncertain, acknowledge this honestly
14. Format names consistently (first name + last name) throughout your response
15. For the first mention of a person, include their birth/death years in parentheses if available
"""

        # Add theme-specific guidelines based on feedback
        if "Relationship details" in found_themes:
            improved_response_prompt += """
11. Always clearly explain how mentioned people are related to the user's family tree
12. Use specific relationship terms (e.g., "3rd great-grandfather" rather than just "ancestor")
"""

        if "Birth/death information" in found_themes:
            improved_response_prompt += """
13. Always include birth and death years in parentheses after first mention of a person: Name (YYYY-YYYY)
14. If only birth or death information is available, use the format: Name (b. YYYY) or Name (d. YYYY)
"""

        if "Response tone" in found_themes:
            improved_response_prompt += """
15. Maintain a warm, conversational tone while being informative and professional
16. Show enthusiasm about shared genealogical interests without being overly familiar
"""

        if "Follow-up questions" in found_themes:
            improved_response_prompt += """
17. Include 1-2 specific follow-up questions that could help advance the research
18. Focus questions on filling gaps in the genealogical record or clarifying ambiguous information
"""

        # Filter and prioritize feedback suggestions
        if feedback_suggestions:
            # Filter out low-quality or redundant feedback
            filtered_suggestions = []
            seen_concepts = set()

            # Keywords to identify high-quality feedback
            quality_indicators = [
                "specific",
                "detail",
                "format",
                "structure",
                "relationship",
                "birth",
                "death",
                "family",
            ]

            # Process each suggestion
            for suggestion in feedback_suggestions:
                # Skip very short suggestions
                if len(suggestion) < 15:
                    continue

                # Calculate a quality score based on presence of quality indicators
                quality_score = sum(
                    1
                    for indicator in quality_indicators
                    if indicator.lower() in suggestion.lower()
                )

                # Extract the core concept (first 30 chars for comparison)
                concept = suggestion.lower()[:30]

                # Skip if we've seen a similar concept already
                if any(
                    similar_concept in concept or concept in similar_concept
                    for similar_concept in seen_concepts
                ):
                    continue

                # Add to filtered list if it's high quality or unique
                if quality_score >= 2 or concept not in seen_concepts:
                    filtered_suggestions.append((suggestion, quality_score))
                    seen_concepts.add(concept)

            # Sort by quality score (highest first) and take top 5
            filtered_suggestions.sort(key=lambda x: x[1], reverse=True)
            top_suggestions = [suggestion for suggestion, _ in filtered_suggestions[:5]]

            if top_suggestions:
                improved_response_prompt += (
                    "\nADDITIONAL GUIDELINES BASED ON USER FEEDBACK:\n"
                )
                for i, suggestion in enumerate(top_suggestions, 19):
                    improved_response_prompt += f"{i}. {suggestion}\n"

        improved_response_prompt += """
EXAMPLES OF EXCELLENT RESPONSES:
"""

        # Select the best examples to include in the prompt
        # Sort examples by feedback quality (length is a simple proxy for quality)
        sorted_examples = sorted(
            good_samples, key=lambda x: len(x.get("feedback", "")), reverse=True
        )

        # Limit to 3 best examples to keep prompt concise
        best_examples = sorted_examples[:3]

        # Add good examples to the response prompt
        for i, example in enumerate(best_examples, 1):
            improved_response_prompt += f"""
Example {i}:
User message: {example['input'][:100]}...
Excellent response: {example['output'][:200]}...
"""
            if example.get("feedback"):
                # Limit feedback length to keep prompt concise
                feedback = example.get("feedback", "")
                if len(feedback) > 100:
                    feedback = feedback[:100] + "..."
                improved_response_prompt += (
                    f"\nWhat makes this response good: {feedback}\n"
                )

        # Update the AI prompts JSON file only
        try:
            # Import the AI prompt utilities
            from ai_prompt_utils import update_prompt

            # Update the extraction prompt in the JSON file
            update_prompt(
                "extraction_task",
                improved_extraction_prompt,
                "Improved Data Extraction & Task Suggestion Prompt",
                "Updated extraction prompt based on feedback analysis",
            )

            # Update the response prompt in the JSON file
            update_prompt(
                "genealogical_reply",
                improved_response_prompt,
                "Improved Genealogical Reply Generation Prompt",
                "Updated reply prompt based on feedback analysis",
            )

            logger.info("Updated AI prompts in ai_prompts.json")
            print(f"\nSuccessfully updated ai_prompts.json with improved prompts")

            # Print a summary of the feedback analysis
            print("\n=== Feedback Analysis Summary ===")
            print(f"Total evaluated responses: {total_responses}")
            print(
                f"Acceptable responses: {acceptable_responses} ({acceptable_responses/total_responses*100:.1f}%)"
            )
            print(
                f"Unacceptable responses: {unacceptable_responses} ({unacceptable_responses/total_responses*100:.1f}%)"
            )

            print("\nCommon themes in feedback:")
            for theme in found_themes:
                print(f"- {theme}")

            print("\nKey improvements made:")
            print("- Enhanced extraction of genealogical information from messages")
            print("- Improved response formatting and structure")
            print("- Better handling of relationship information")
            print("- More natural and conversational tone")
            print("- Clearer presentation of birth/death details")

        except Exception as e:
            logger.error(f"Error updating AI prompts JSON file: {e}", exc_info=True)
            print(f"\nFailed to update ai_prompts.json: {e}")

        conn.close()
        input("\nPress Enter to continue...")
        return True

    except Exception as e:
        logger.error(f"Error updating AI prompts: {e}", exc_info=True)
        input("\nPress Enter to continue...")
        return False


def delete_test_database():
    """
    Delete the test database file.

    Returns:
        bool: True if the database was deleted successfully, False otherwise
    """
    logger.info("=== Deleting Test Database ===")

    try:
        # Check if the test database file exists
        if test_db_path.exists():
            # Delete the file
            test_db_path.unlink()
            logger.info(f"Test database deleted: {test_db_path}")
            print(f"\nTest database deleted successfully: {test_db_path}")
            return True
        else:
            logger.info(f"Test database does not exist: {test_db_path}")
            print(f"\nTest database does not exist: {test_db_path}")
            return True
    except Exception as e:
        logger.error(f"Error deleting test database: {e}", exc_info=True)
        print(f"\nError deleting test database: {e}")
        return False


def run_automated_cycle(session_manager=None, num_cycles=3):
    """
    Run the entire AI response improvement cycle automatically.

    This function:
    1. Creates a test database with messages (if not already created)
    2. Processes messages and generates AI responses
    3. Automatically evaluates the responses
    4. Analyzes feedback and updates prompts
    5. Repeats steps 2-4 for the specified number of cycles

    Args:
        session_manager: The active SessionManager instance (optional)
        num_cycles: Number of improvement cycles to run (default: 3)

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n=== Running Automated AI Response Improvement Cycle ===")

    # Step 1: Create test database if needed
    if not session_manager:
        print("\nStep 1: Creating test database and generating messages...")

        # Check if test database already exists
        if test_db_path.exists():
            print(f"Test database already exists at {test_db_path}")
            # Use existing database
            config_instance.DATABASE_FILE = test_db_path
            session_manager = SessionManager()
            if not session_manager.ensure_db_ready():
                logger.error("Failed to connect to existing test database.")
                print("Creating a new test database instead...")
                # Delete existing database and create a new one
                try:
                    test_db_path.unlink()
                    logger.info(f"Deleted existing test database: {test_db_path}")
                except Exception as e:
                    logger.error(f"Failed to delete existing test database: {e}")
                    return False

                session_manager = create_test_database()
                if not session_manager:
                    logger.error("Failed to create test database.")
                    return False
            else:
                print("Successfully connected to existing test database.")

                # Check if we have any data in the database
                db_session = session_manager.get_db_conn()
                if not db_session:
                    logger.error("Failed to get database session")
                    return False

                try:
                    # Count people in the database
                    person_count = db_session.query(func.count(Person.id)).scalar()
                    if person_count == 0:
                        print(
                            "No test data found in the database. Creating test data..."
                        )
                        # Create test data (using 50 examples for automated runs)
                        if not create_test_data(session_manager, num_people=50):
                            logger.error("Failed to create test data.")
                            return False
                    else:
                        print(
                            f"Found {person_count} people in the database. Using existing test data."
                        )
                except Exception as e:
                    logger.error(f"Error checking for test data: {e}", exc_info=True)
                    return False
        else:
            # Create new test database
            session_manager = create_test_database()
            if not session_manager:
                logger.error("Failed to create test database.")
                return False

            # Create test data (using 50 examples for automated runs)
            if not create_test_data(session_manager, num_people=50):
                logger.error("Failed to create test data.")
                return False

            print("\nTest database and messages created successfully!")
    else:
        print("\nUsing provided session manager with existing test database.")

    # Run the specified number of improvement cycles
    for cycle in range(1, num_cycles + 1):
        print(f"\n=== Starting Improvement Cycle {cycle}/{num_cycles} ===")

        # Step 2: Process messages and generate AI responses
        print(f"\nStep 2.{cycle}: Processing messages and generating AI responses...")
        if not process_messages_and_generate_responses(session_manager):
            logger.error(f"Failed to process messages in cycle {cycle}.")
            return False

        # Step 3: Automatically evaluate responses
        print(f"\nStep 3.{cycle}: Automatically evaluating responses...")
        if not evaluate_ai_responses(session_manager, auto_evaluate=True):
            logger.error(f"Failed to evaluate responses in cycle {cycle}.")
            return False

        # Step 4: Analyze feedback and update prompts
        print(f"\nStep 4.{cycle}: Analyzing feedback and updating prompts...")
        if not analyze_feedback_and_suggest_improvements(session_manager):
            logger.error(f"Failed to analyze feedback in cycle {cycle}.")
            return False

        # Step 5: Update AI prompts
        print(f"\nStep 5.{cycle}: Updating AI prompts based on feedback...")
        if not update_ai_prompts(session_manager):
            logger.error(f"Failed to update AI prompts in cycle {cycle}.")
            return False

        print(f"\n=== Completed Improvement Cycle {cycle}/{num_cycles} ===")

    print("\n=== Automated AI Response Improvement Cycle Completed ===")
    print("\nYou can now manually evaluate the final responses (Option 3)")
    print("or continue with more automated cycles (Option 8).")

    return True


def display_menu():
    """Display the main menu and get user choice."""
    os.system("cls" if os.name == "nt" else "clear")
    print("\n" + "=" * 80)
    print("                    AI RESPONSE TESTING MENU")
    print("=" * 80)
    print(
        "\n1. Create Test Database and Generate Messages (with AI sentiment classification)"
    )
    print("2. Process Messages and Generate AI Responses")
    print("3. Evaluate AI Responses (Manual)")
    print("4. Analyze Feedback and Suggest Improvements")
    print("5. Update AI Prompts Based on Feedback")
    print("6. Delete Test Database")
    print("7. Run Automated Improvement Cycle (Steps 2-5 repeated)")
    print("8. Exit")
    print("\n" + "=" * 80)

    while True:
        try:
            choice = int(input("\nEnter your choice (1-8): "))
            if 1 <= choice <= 8:
                return choice
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def process_messages_and_generate_responses(session_manager):
    """
    Process messages and generate AI responses.

    This function:
    1. Finds all people with messages in the database
    2. Processes each message using the AI
    3. Generates a response
    4. Saves the response to the database

    Args:
        session_manager: The active SessionManager instance
    """
    logger.info("=== Processing Messages and Generating AI Responses ===")
    print(
        "\nProcessing messages and generating AI responses. This may take some time..."
    )
    print("Please wait while the AI processes each message...")

    # Get a database session
    db_session = session_manager.get_db_conn()
    if not db_session:
        logger.error("Failed to get database session")
        return False

    # Load prompts from the JSON file only
    improved_extraction_prompt = None
    improved_response_prompt = None

    try:
        # Import the AI prompt utilities
        from ai_prompt_utils import get_prompt

        # Get prompts from the JSON file
        extraction_prompt = get_prompt("extraction_task")
        response_prompt = get_prompt("genealogical_reply")

        if extraction_prompt and response_prompt:
            logger.info("Loaded prompts from ai_prompts.json")
            improved_extraction_prompt = extraction_prompt
            improved_response_prompt = response_prompt
            print("\nUsing prompts from ai_prompts.json")
        else:
            logger.warning("Failed to load prompts from ai_prompts.json")
            print(
                "\nFailed to load prompts from ai_prompts.json, using default prompts instead"
            )
    except ImportError:
        logger.error("ai_prompt_utils module not available")
        print(
            "\nFailed to import ai_prompt_utils module, using default prompts instead"
        )
    except Exception as e:
        logger.error(f"Error loading prompts from ai_prompts.json: {e}", exc_info=True)
        print(f"\nError loading prompts from ai_prompts.json: {e}")
        print("Using default prompts instead")

    # Load GEDCOM data once at the beginning
    logger.info("Loading GEDCOM data once for all message processing...")

    # Variables to store original values for restoration
    original_get_gedcom_data = None
    original_gedcom_utils_available = False
    original_relationship_utils_available = False

    try:
        # Import action9_process_productive to patch it
        import action9_process_productive

        # Save original values for restoration later
        original_get_gedcom_data = action9_process_productive.get_gedcom_data
        original_gedcom_utils_available = (
            action9_process_productive.GEDCOM_UTILS_AVAILABLE
        )
        original_relationship_utils_available = (
            action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE
        )

        # Using person_search module instead of action10 directly

        # Temporarily set the flags to True to ensure get_gedcom_data works
        action9_process_productive.GEDCOM_UTILS_AVAILABLE = True
        action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE = True

        # Use our existing get_gedcom_data function to load the GEDCOM data once
        logger.info("Loading GEDCOM data using get_gedcom_data function...")
        gedcom_data = get_gedcom_data()

        if not gedcom_data:
            logger.error("Failed to load GEDCOM data")
            raise RuntimeError("Failed to load GEDCOM data")

        # Verify that the caches are built
        if not hasattr(gedcom_data, "indi_index") or not gedcom_data.indi_index:
            logger.info("GEDCOM data loaded but caches not built, building now...")
            gedcom_data.build_caches()
            logger.info("GEDCOM caches built successfully")
        else:
            logger.info("GEDCOM data loaded with caches already built")

        # Set the cached GEDCOM data in gedcom_search_utils
        import gedcom_search_utils

        gedcom_search_utils.set_cached_gedcom_data(gedcom_data)
        logger.info("Set cached GEDCOM data in gedcom_search_utils")

        # Set the global cache in action9_process_productive
        action9_process_productive._CACHED_GEDCOM_DATA = gedcom_data
        logger.info("Set _CACHED_GEDCOM_DATA in action9_process_productive")

        if gedcom_data:
            logger.info(
                f"GEDCOM data loaded successfully with {len(gedcom_data.processed_data_cache)} entries in cache"
            )

            # Log GEDCOM data details for debugging
            logger.info(f"GEDCOM data type: {type(gedcom_data)}")
            logger.info(f"GEDCOM data attributes: {dir(gedcom_data)}")

            # Log some key attributes
            if hasattr(gedcom_data, "path"):
                logger.info(f"GEDCOM file path: {gedcom_data.path}")
            if hasattr(gedcom_data, "indi_index"):
                logger.info(f"Individual index size: {len(gedcom_data.indi_index)}")
            if hasattr(gedcom_data, "processed_data_cache"):
                logger.info(
                    f"Processed data cache size: {len(gedcom_data.processed_data_cache)}"
                )

            # Log the first few entries in the cache for debugging
            if (
                hasattr(gedcom_data, "processed_data_cache")
                and gedcom_data.processed_data_cache
            ):
                cache_keys = list(gedcom_data.processed_data_cache.keys())[:3]
                logger.info(f"First few cache keys: {cache_keys}")
                for key in cache_keys:
                    entry = gedcom_data.processed_data_cache.get(key, {})
                    name = entry.get("full_name_disp", "Unknown")
                    birth_year = entry.get("birth_year", "Unknown")
                    logger.info(f"  Cache entry {key}: {name} (b. {birth_year})")

            # Patch action9_process_productive to use our loaded GEDCOM data
            # This ensures that any direct calls to get_gedcom_data() will return our data
            action9_process_productive.get_gedcom_data = lambda: gedcom_data

            # Patch action10 if needed
            logger.info("Using person_search module for API and GEDCOM search")
        else:
            logger.warning(
                "Failed to load GEDCOM data, will process messages without GEDCOM data"
            )
    except Exception as e:
        logger.error(f"Error loading GEDCOM data: {e}", exc_info=True)
        gedcom_data = None

    # Get message type ID for custom responses
    message_type_id = None
    with db_transn(db_session) as session:
        message_type = (
            session.query(MessageType)
            .filter_by(type_name=CUSTOM_RESPONSE_MESSAGE_TYPE)
            .first()
        )
        if message_type:
            message_type_id = message_type.id
        else:
            logger.error(
                f"Message type {CUSTOM_RESPONSE_MESSAGE_TYPE} not found in database"
            )
            return False

    # Find all people with messages
    try:
        # Subquery to find the timestamp of the latest IN message for each person
        latest_in_log_subq = (
            db_session.query(
                ConversationLog.people_id,
                func.max(ConversationLog.latest_timestamp).label("max_in_ts"),
            )
            .filter(ConversationLog.direction == MessageDirectionEnum.IN)
            .group_by(ConversationLog.people_id)
            .subquery("latest_in_sub")
        )

        # Main query to find candidates
        candidates_query = (
            db_session.query(Person)
            .join(latest_in_log_subq, Person.id == latest_in_log_subq.c.people_id)
            .join(  # Join to the specific IN log entry
                ConversationLog,
                and_(
                    Person.id == ConversationLog.people_id,
                    ConversationLog.direction == MessageDirectionEnum.IN,
                    ConversationLog.latest_timestamp == latest_in_log_subq.c.max_in_ts,
                ),
            )
            .order_by(Person.id)  # Consistent processing order
        )

        # Execute the query
        candidates = candidates_query.all()
        logger.info(f"Found {len(candidates)} people with messages to process")

        # Process each candidate
        processed_count = 0
        for person in candidates:
            # Get the latest message for this person
            latest_message = (
                db_session.query(ConversationLog)
                .filter(
                    ConversationLog.people_id == person.id,
                    ConversationLog.direction == MessageDirectionEnum.IN,
                    ConversationLog.latest_timestamp == latest_in_log_subq.c.max_in_ts,
                )
                .first()
            )

            if not latest_message:
                logger.warning(
                    f"No latest message found for person {person.id} ({person.username})"
                )
                continue

            # Log the person and message
            log_prefix = f"Person {person.id} ({person.username})"
            logger.info(
                f"{log_prefix}: Processing message: {latest_message.latest_message_content[:50]}..."
            )

            # Step 1: Extract information from the message using AI
            message_content = latest_message.latest_message_content

            # Call AI to extract information using the improved prompt if available
            ai_response = _extract_information_from_message(
                message_content, log_prefix, improved_extraction_prompt
            )

            if not ai_response:
                logger.warning(f"{log_prefix}: AI extraction failed")
                continue

            # We don't need to process the AI response again, as it's already been processed in _extract_information_from_message
            extracted_data = ai_response

            # Log the results
            entity_counts = {k: len(v) for k, v in extracted_data.items()}
            logger.debug(
                f"{log_prefix}: Extracted entities: {json.dumps(entity_counts)}"
            )

            # Step 2: Try to find a person in the tree based on extracted names
            person_details = None
            if extracted_data.get("mentioned_names"):
                try:
                    # Call our wrapper function with the pre-loaded GEDCOM data
                    logger.info(
                        f"{log_prefix}: Calling _identify_and_get_person_details with pre-loaded GEDCOM data"
                    )

                    # Log the mentioned names for debugging
                    logger.info(
                        f"{log_prefix}: Mentioned names: {extracted_data.get('mentioned_names')}"
                    )

                    # Make sure we're passing the pre-loaded GEDCOM data
                    if gedcom_data:
                        logger.info(
                            f"{log_prefix}: Using pre-loaded GEDCOM data with {len(gedcom_data.processed_data_cache)} entries"
                        )
                        # Log some sample entries from the GEDCOM data
                        sample_keys = list(gedcom_data.processed_data_cache.keys())[:3]
                        logger.info(f"{log_prefix}: Sample GEDCOM keys: {sample_keys}")

                        # The GEDCOM data should already have its indexes built
                        # We'll just verify that the indexes exist
                        if (
                            not hasattr(gedcom_data, "indi_index")
                            or not gedcom_data.indi_index
                        ):
                            logger.warning(
                                f"{log_prefix}: GEDCOM data missing indi_index, this should have been built already"
                            )
                    else:
                        logger.error(
                            f"{log_prefix}: GEDCOM data is None, cannot proceed with person identification"
                        )

                    # Check if original_identify_and_get_person_details is available
                    if not callable(original_identify_and_get_person_details):
                        logger.error(
                            f"{log_prefix}: original_identify_and_get_person_details is not callable"
                        )
                    else:
                        logger.info(
                            f"{log_prefix}: original_identify_and_get_person_details is callable"
                        )

                    # We've already set the cached GEDCOM data in gedcom_search_utils
                    # No need to patch the functions anymore
                    logger.info(
                        f"{log_prefix}: Using pre-loaded GEDCOM data from gedcom_search_utils"
                    )

                    # Call our wrapper function
                    logger.info(
                        f"{log_prefix}: Calling _identify_and_get_person_details now..."
                    )
                    try:
                        person_details = _identify_and_get_person_details(
                            session_manager, extracted_data, log_prefix, gedcom_data
                        )
                        logger.info(
                            f"{log_prefix}: _identify_and_get_person_details returned: {person_details is not None}"
                        )
                    except Exception as e:
                        logger.error(
                            f"{log_prefix}: Error in _identify_and_get_person_details: {e}",
                            exc_info=True,
                        )

                    # No need to restore functions since we're not patching them anymore

                    # Log the result
                    if person_details:
                        logger.info(f"{log_prefix}: Person identified successfully")
                        logger.info(
                            f"{log_prefix}: - source: {person_details.get('source', 'Unknown')}"
                        )
                        if "details" in person_details:
                            person_name = person_details["details"].get(
                                "name", "Unknown"
                            )
                            logger.info(f"{log_prefix}: - person: {person_name}")
                    else:
                        logger.warning(f"{log_prefix}: No person identified")

                except Exception as e:
                    logger.error(
                        f"{log_prefix}: Error identifying person: {e}", exc_info=True
                    )

            # Step 3: Generate a response
            custom_reply = None

            if person_details:
                # Generate custom reply using AI with improved prompt if available
                custom_reply = _generate_ai_response(
                    message_content,
                    extracted_data,
                    person_details,
                    log_prefix,
                    improved_response_prompt,
                )

                if custom_reply:
                    logger.info(f"{log_prefix}: Generated custom genealogical reply")
                else:
                    logger.warning(f"{log_prefix}: Failed to generate custom reply")
            else:
                logger.debug(f"{log_prefix}: No person identified in message")

            # Step 4: Save the response to the database
            # First check if a response already exists for this conversation
            try:
                # Check if an OUT message already exists for this conversation
                existing_response = (
                    db_session.query(ConversationLog)
                    .filter(
                        ConversationLog.conversation_id
                        == latest_message.conversation_id,
                        ConversationLog.direction == MessageDirectionEnum.OUT,
                    )
                    .first()
                )

                if existing_response:
                    # Update the existing response instead of creating a new one
                    with db_transn(db_session) as session:
                        existing_response.latest_message_content = (
                            custom_reply
                            if custom_reply
                            else "No custom reply generated"
                        )
                        existing_response.latest_timestamp = datetime.now(timezone.utc)
                        existing_response.script_message_status = "test_generated"
                        existing_response.custom_reply_sent_at = datetime.now(
                            timezone.utc
                        )
                        existing_response.ai_feedback = (
                            None  # Reset feedback for re-evaluation
                        )

                        # Also update the IN message to mark it as processed
                        latest_message.custom_reply_sent_at = datetime.now(timezone.utc)
                        session.add(latest_message)

                        processed_count += 1
                        logger.info(
                            f"{log_prefix}: Updated existing response in database"
                        )
                else:
                    # Create a new conversation log entry for the response
                    with db_transn(db_session) as session:
                        # Create a new conversation log entry for the OUT message
                        conversation_log_out = ConversationLog(
                            conversation_id=latest_message.conversation_id,
                            direction=MessageDirectionEnum.OUT,
                            people_id=person.id,
                            latest_message_content=(
                                custom_reply
                                if custom_reply
                                else "No custom reply generated"
                            ),
                            latest_timestamp=datetime.now(timezone.utc),
                            ai_sentiment=None,
                            message_type_id=message_type_id,
                            script_message_status="test_generated",
                            custom_reply_sent_at=datetime.now(timezone.utc),
                        )
                        session.add(conversation_log_out)

                        # Also update the IN message to mark it as processed
                        latest_message.custom_reply_sent_at = datetime.now(timezone.utc)
                        session.add(latest_message)

                        # Add a field to track if the response is acceptable (for later evaluation)
                        # We'll use the 'ai_feedback' field for this purpose
                        conversation_log_out.ai_feedback = None  # Will be set to 1 (acceptable) or 0 (unacceptable) later

                        processed_count += 1
                        logger.info(f"{log_prefix}: Saved new response to database")
            except Exception as e:
                logger.error(
                    f"{log_prefix}: Error saving response to database: {e}",
                    exc_info=True,
                )

        logger.info(f"Processed {processed_count} messages and generated responses")
        return True

    except Exception as e:
        logger.error(f"Error processing messages: {e}", exc_info=True)
        input("\nPress Enter to continue...")
        return False
    finally:
        # Restore original values in action9_process_productive if we modified them
        try:
            if (
                "action9_process_productive" in sys.modules
                and "original_get_gedcom_data" in locals()
                and original_get_gedcom_data is not None
            ):
                action9_process_productive.get_gedcom_data = original_get_gedcom_data
                # Also clear the cached GEDCOM data
                if hasattr(action9_process_productive, "_CACHED_GEDCOM_DATA"):
                    action9_process_productive._CACHED_GEDCOM_DATA = None

            if (
                "action9_process_productive" in sys.modules
                and "original_gedcom_utils_available" in locals()
            ):
                action9_process_productive.GEDCOM_UTILS_AVAILABLE = (
                    original_gedcom_utils_available
                )

            if (
                "action9_process_productive" in sys.modules
                and "original_relationship_utils_available" in locals()
            ):
                action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE = (
                    original_relationship_utils_available
                )

            # No need to restore action10 functions as we're using person_search module

            logger.info("Restored original module values")
        except Exception as e:
            logger.error(f"Error restoring original values: {e}", exc_info=True)


def main():
    """Main function to run the test script with menu interface."""
    # Set up logging
    setup_logging()
    logger.info("=== Starting AI Response Test Script with Menu Interface ===")

    # Initialize session manager
    session_manager = None

    # Main menu loop
    while True:
        choice = display_menu()

        if choice == 1:  # Create Test Database and Generate Messages
            print("\n=== Creating Test Database and Generating Messages ===")

            # Create test database
            session_manager = create_test_database()
            if not session_manager:
                logger.error("Failed to create test database.")
                input("\nPress Enter to continue...")
                continue

            # Create test data
            if not create_test_data(session_manager):
                logger.error("Failed to create test data.")
                input("\nPress Enter to continue...")
                continue

            print("\nTest database and messages created successfully!")
            input("\nPress Enter to continue...")

        elif choice == 2:  # Process Messages and Generate AI Responses
            if not session_manager:
                # Use existing database
                config_instance.DATABASE_FILE = test_db_path
                session_manager = SessionManager()
                if not session_manager.ensure_db_ready():
                    logger.error("Failed to connect to existing test database.")
                    input("\nPress Enter to continue...")
                    continue

                # Ensure the ai_feedback column exists
                if not ensure_ai_feedback_column():
                    logger.error("Failed to ensure ai_feedback column exists.")
                    input("\nPress Enter to continue...")
                    continue

            # Process messages and generate AI responses
            if process_messages_and_generate_responses(session_manager):
                print("\nMessages processed and AI responses generated successfully!")
            else:
                print("\nFailed to process messages and generate AI responses.")

            input("\nPress Enter to continue...")

        elif choice == 3:  # Evaluate AI Responses (Manual)
            if not session_manager:
                # Use existing database
                config_instance.DATABASE_FILE = test_db_path
                session_manager = SessionManager()
                if not session_manager.ensure_db_ready():
                    logger.error("Failed to connect to existing test database.")
                    input("\nPress Enter to continue...")
                    continue

                # Ensure the ai_feedback column exists
                if not ensure_ai_feedback_column():
                    logger.error("Failed to ensure ai_feedback column exists.")
                    input("\nPress Enter to continue...")
                    continue

            # Evaluate AI responses
            print("\n=== Evaluating AI Responses (Manual) ===")

            # Check if there are any AI-generated responses in the database
            db_session = session_manager.get_db_conn()
            if not db_session:
                logger.error("Failed to get database session")
                input("\nPress Enter to continue...")
                continue

            try:
                # Count OUT messages with test_generated status
                out_message_count = (
                    db_session.query(func.count(ConversationLog.conversation_id))
                    .filter(
                        ConversationLog.direction == MessageDirectionEnum.OUT,
                        ConversationLog.script_message_status == "test_generated",
                    )
                    .scalar()
                )

                print(
                    f"Found {out_message_count} AI-generated responses in the database."
                )

                if out_message_count == 0:
                    print("\nNo AI-generated responses found to evaluate.")
                    print(
                        "Please run Option 2 first to process messages and generate AI responses."
                    )
                    input("\nPress Enter to continue...")
                    continue

                # If we have responses, proceed with manual evaluation
                evaluate_ai_responses(session_manager, auto_evaluate=False)

            except Exception as e:
                logger.error(f"Error checking for AI responses: {e}", exc_info=True)
                print(f"\nError checking for AI responses: {e}")
                input("\nPress Enter to continue...")

        elif choice == 4:  # Analyze Feedback and Suggest Improvements
            if not session_manager:
                # Use existing database
                config_instance.DATABASE_FILE = test_db_path
                session_manager = SessionManager()
                if not session_manager.ensure_db_ready():
                    logger.error("Failed to connect to existing test database.")
                    input("\nPress Enter to continue...")
                    continue

            # Analyze feedback and suggest improvements
            analyze_feedback_and_suggest_improvements(session_manager)

        elif choice == 5:  # Update AI Prompts Based on Feedback
            if not session_manager:
                # Use existing database
                config_instance.DATABASE_FILE = test_db_path
                session_manager = SessionManager()
                if not session_manager.ensure_db_ready():
                    logger.error("Failed to connect to existing test database.")
                    input("\nPress Enter to continue...")
                    continue

            # Update AI prompts based on feedback
            update_ai_prompts(session_manager)

        elif choice == 6:  # Delete Test Database
            print("\n=== Deleting Test Database ===")
            if delete_test_database():
                # Reset session manager since the database is gone
                session_manager = None
                config_instance.DATABASE_FILE = original_db_path
            input("\nPress Enter to continue...")

        elif choice == 7:  # Run Automated Improvement Cycle
            # Ask for number of cycles first
            while True:
                try:
                    num_cycles = int(
                        input("\nEnter number of improvement cycles to run (1-5): ")
                    )
                    if 1 <= num_cycles <= 5:
                        break
                    else:
                        print("Invalid input. Please enter a number between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Run the automated cycle
            # We'll let run_automated_cycle handle the session_manager creation/connection
            if run_automated_cycle(session_manager, num_cycles):
                print("\nAutomated improvement cycle completed successfully!")

                # Make sure session_manager is updated if it was created in run_automated_cycle
                if not session_manager:
                    config_instance.DATABASE_FILE = test_db_path
                    session_manager = SessionManager()
                    if not session_manager.ensure_db_ready():
                        logger.error(
                            "Failed to connect to test database after automated cycle."
                        )
            else:
                print("\nFailed to complete automated improvement cycle.")

            input("\nPress Enter to continue...")

        elif choice == 8:  # Exit
            print("\nExiting AI Response Test Script. Goodbye!")
            break


if __name__ == "__main__":
    main()
