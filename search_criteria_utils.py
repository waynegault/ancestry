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
    gender = _parse_gender_input(gender_input) if gender_input else None

    # Birth year
    birth_year = _parse_year_input(get_input_func("  Birth Year (YYYY): "))

    # Birth place
    birth_place = _sanitize_input(get_input_func("  Birth Place Contains: "))

    # Death year (optional)
    death_year = _parse_year_input(get_input_func("  Death Year (YYYY) [Optional]: "))

    # Death place (optional)
    death_place = _sanitize_input(get_input_func("  Death Place Contains [Optional]: "))

    # Create date objects
    birth_date_obj = _create_date_object(birth_year, "birth")
    death_date_obj = _create_date_object(death_year, "death")

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


def _parse_year_input(year_str: str) -> Optional[int]:
    """Parse year input string to integer."""
    return int(year_str) if year_str.strip().isdigit() else None


def _create_date_object(year: Optional[int], date_type: str) -> Optional[datetime]:
    """Create datetime object from year, with error handling."""
    if not year:
        return None
    try:
        return datetime(year, 1, 1, tzinfo=timezone.utc)
    except ValueError:
        logger.warning(f"Cannot create date object for {date_type} year {year}.")
        return None


def _parse_gender_input(gender_input: str) -> Optional[str]:
    """Parse gender input to standardized format."""
    if gender_input and gender_input[0].lower() in ["m", "f"]:
        return gender_input[0].lower()
    return None


def _format_years_display(birth_year: Optional[int], death_year: Optional[int]) -> str:
    """Format birth and death years for display."""
    if birth_year and death_year:
        return f" ({birth_year}-{death_year})"
    if birth_year:
        return f" (b. {birth_year})"
    if death_year:
        return f" (d. {death_year})"
    return ""


def _print_section_header(label: str, is_first: bool) -> None:
    """Print section header with appropriate spacing."""
    if is_first:
        print(f"{label}:")
    else:
        print(f"\n{label}:")


def _print_family_member(member: dict) -> None:
    """Print a single family member's information."""
    name = member.get("name", "Unknown")
    birth_year = member.get("birth_year")
    death_year = member.get("death_year")
    years_display = _format_years_display(birth_year, death_year)
    print(f"   - {name}{years_display}")


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

        _print_section_header(label, first_section)
        first_section = False

        if not members:
            print("   - None found")
            continue

        for member in members:
            if member:
                _print_family_member(member)

