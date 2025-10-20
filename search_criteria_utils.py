#!/usr/bin/env python3
"""
Unified search criteria collection for genealogical research actions.

Provides consistent user interaction for both Action 10 (GEDCOM) and Action 11 (API)
to ensure identical search criteria collection and validation.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def get_unified_search_criteria(
    get_input_func: Optional[Callable[[str], str]] = None,
) -> Optional[dict[str, Any]]:
    """
    Collect unified search criteria from user input.
    
    This function provides consistent search criteria collection for both
    Action 10 (GEDCOM) and Action 11 (API) to ensure identical user experience.
    
    Args:
        get_input_func: Optional function to get user input (for testing).
                       If None, uses built-in input().
    
    Returns:
        Dictionary with standardized search criteria, or None if cancelled.
    """
    if get_input_func is None:
        get_input_func = input

    print("--- Search Criteria ---\n")

    # Collect basic criteria
    first_name = _sanitize_input(get_input_func("  First Name Contains: "))
    surname = _sanitize_input(get_input_func("  Surname Contains: "))

    # Validate required fields
    if not (first_name or surname):
        logger.warning("Search requires First Name or Surname. Search cancelled.")
        print("\nSearch requires First Name or Surname. Search cancelled.")
        return None

    # Gender
    gender_input = _sanitize_input(get_input_func("  Gender (M/F): "))
    gender = None
    if gender_input and gender_input[0].lower() in ["m", "f"]:
        gender = gender_input[0].lower()

    # Birth year
    birth_year_str = get_input_func("  Birth Year (YYYY): ").strip()
    birth_year = int(birth_year_str) if birth_year_str.isdigit() else None

    # Birth place
    birth_place = _sanitize_input(get_input_func("  Birth Place Contains: "))

    # Death year (optional)
    death_year_str = get_input_func("  Death Year (YYYY) [Optional]: ").strip()
    death_year = int(death_year_str) if death_year_str.isdigit() else None

    # Death place (optional)
    death_place = _sanitize_input(get_input_func("  Death Place Contains [Optional]: "))

    # Create date objects
    birth_date_obj = None
    if birth_year:
        try:
            birth_date_obj = datetime(birth_year, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Cannot create date object for birth year {birth_year}.")
            birth_year = None

    death_date_obj = None
    if death_year:
        try:
            death_date_obj = datetime(death_year, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Cannot create date object for death year {death_year}.")
            death_year = None

    # Build standardized criteria dictionary
    criteria = {
        "first_name": first_name,
        "surname": surname,
        "gender": gender,
        "birth_year": birth_year,
        "birth_date_obj": birth_date_obj,
        "birth_place": birth_place,
        "death_year": death_year,
        "death_date_obj": death_date_obj,
        "death_place": death_place,
    }

    # Log criteria
    logger.debug("\n--- Search Criteria Collected ---")
    for key, value in criteria.items():
        if value is not None and key not in ["birth_date_obj", "death_date_obj"]:
            logger.debug(f"  {key.replace('_', ' ').title()}: {value}")

    return criteria


def _sanitize_input(value: str) -> Optional[str]:
    """Sanitize user input by stripping whitespace."""
    if not value:
        return None
    sanitized = value.strip()
    return sanitized if sanitized else None


def display_family_members(
    family_data: dict[str, list],
    relation_labels: Optional[dict[str, str]] = None
) -> None:
    """
    Display family members in a consistent format for both Action 10 and Action 11.

    Args:
        family_data: Dictionary with keys like 'parents', 'siblings', 'spouses', 'children'
                    and values as lists of family member dictionaries
        relation_labels: Optional custom labels for each relation type
    """
    if relation_labels is None:
        relation_labels = {
            "parents": "📋 Parents",
            "siblings": "📋 Siblings",
            "spouses": "💕 Spouses",
            "children": "👶 Children",
        }

    first_section = True
    for relation_key, label in relation_labels.items():
        members = family_data.get(relation_key, [])

        # Add blank line before each section except the first
        if first_section:
            print(f"{label}:")
            first_section = False
        else:
            print(f"\n{label}:")

        if not members:
            print("   - None found")
            continue

        for member in members:
            if not member:
                continue

            # Extract member information
            name = member.get("name", "Unknown")
            birth_year = member.get("birth_year")
            death_year = member.get("death_year")

            # Format years display
            years_display = ""
            if birth_year and death_year:
                years_display = f" ({birth_year}-{death_year})"
            elif birth_year:
                years_display = f" (b. {birth_year})"
            elif death_year:
                years_display = f" (d. {death_year})"

            print(f"   - {name}{years_display}")

