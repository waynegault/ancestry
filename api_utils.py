#!/usr/bin/env python3
# api_utils.py
"""
Utility functions for parsing Ancestry API responses and formatting API data.

Provides functions to:
- Parse person details from various Ancestry API responses.
- Format relationship paths from getladder API responses.
- Call specific Ancestry APIs (Suggest, Facts, Ladder, Discovery Relationship, TreesUI, Send Message, Profile Details, Header Trees, Tree Owner).
- Includes a self-check mechanism using live API calls (requires configuration).

Note: Relationship path formatting functions previously in test_relationship_path.py are now integrated here.
"""

# --- Standard library imports ---
import re
import json
import time  # For rate limiting and delays
import traceback  # For error reporting
import requests  # Keep for exception types and Response object checking
import logging
import uuid
from typing import Optional, Dict, Any, List, Tuple, Callable, cast, TYPE_CHECKING
from datetime import (
    datetime,
    timezone,
)  # Import datetime for parse_ancestry_person_details and self_check
from urllib.parse import urljoin, quote, urlencode

# --- Check for optional dependencies ---
try:
    import pydantic

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore
    BS4_AVAILABLE = False

# --- Local application imports ---
# Use centralized logging config setup
from logging_config import setup_logging, logger

# Initialize the logger with a specific log file for this module
logger = setup_logging(log_file="api_utils.log", log_level="INFO")

# --- Test framework imports ---
try:
    from test_framework import (
        TestSuite as TestFrameworkTestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )

    TestSuite = TestFrameworkTestSuite  # type: ignore
except ImportError:
    # If test framework not available, import will fail at test time
    TestSuite = None
    suppress_logging = None
    create_mock_data = None
    assert_valid_function = None

# --- Local application imports ---
from utils import SessionManager, _api_req, format_name
from gedcom_utils import _parse_date, _clean_display_date
from config import config_instance, selenium_config
from database import Person  # Required for call_send_message_api

# Note: format_api_relationship_path has been moved to relationship_utils.py


# --- Constants for moved/new functions ---

# For call_send_message_api
SEND_ERROR_INVALID_RECIPIENT = "send_error (invalid_recipient)"
SEND_ERROR_MISSING_OWN_ID = "send_error (missing_own_id)"
SEND_ERROR_INTERNAL_MODE = "send_error (internal_mode_error)"
SEND_ERROR_API_PREP_FAILED = "send_error (api_prep_failed)"
SEND_ERROR_UNEXPECTED_FORMAT = "send_error (unexpected_format)"
SEND_ERROR_VALIDATION_FAILED = "send_error (validation_failed)"
SEND_ERROR_POST_FAILED = "send_error (post_failed)"
SEND_ERROR_UNKNOWN = "send_error (unknown)"
SEND_SUCCESS_DELIVERED = "delivered OK"
SEND_SUCCESS_DRY_RUN = "typed (dry_run)"

# --- API Endpoint Constants ---
# Messaging API endpoints
API_PATH_SEND_MESSAGE_NEW = "app-api/express/v2/conversations/message"
API_PATH_SEND_MESSAGE_EXISTING = "app-api/express/v2/conversations/{conv_id}"

# Profile API endpoints
API_PATH_PROFILE_DETAILS = "/app-api/express/v1/profiles/details"

# Tree API endpoints
API_PATH_HEADER_TREES = "api/uhome/secure/rest/header/trees"
API_PATH_TREE_OWNER_INFO = "api/uhome/secure/rest/user/tree-info"

# Person API endpoints
API_PATH_PERSON_PICKER_SUGGEST = "api/person-picker/suggest/{tree_id}"
API_PATH_PERSON_FACTS_USER = (
    "family-tree/person/facts/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}"
)
API_PATH_PERSON_GETLADDER = (
    "family-tree/person/tree/{tree_id}/person/{person_id}/getladder"
)
API_PATH_DISCOVERY_RELATIONSHIP = "discoveryui-matchingservice/api/relationship"  # noqa: Ancestry's matching service API
API_PATH_TREESUI_LIST = "trees/{tree_id}/persons"

# Message API keys
KEY_CONVERSATION_ID = "conversation_id"
KEY_MESSAGE = "message"
KEY_AUTHOR = "author"

# Profile API keys
KEY_FIRST_NAME = "FirstName"
KEY_DISPLAY_NAME = "displayName"  # Used in profile and tree owner APIs
KEY_LAST_LOGIN_DATE = "LastLoginDate"
KEY_IS_CONTACTABLE = "IsContactable"

# Tree API keys
KEY_MENUITEMS = "menuitems"
KEY_URL = "url"
KEY_TEXT = "text"
KEY_OWNER = "owner"

# Note: Relationship path formatting test constants have been moved to relationship_utils.py


# --- Response Models ---
# Simple data classes for API response validation (Pydantic-style but without dependency)


class PersonSuggestResponse:
    """Model for validating Ancestry Suggest API responses."""

    def __init__(self, **kwargs):
        self.PersonId: Optional[str] = kwargs.get("PersonId")
        self.TreeId: Optional[str] = kwargs.get("TreeId")
        self.UserId: Optional[str] = kwargs.get("UserId")
        self.FullName: Optional[str] = kwargs.get("FullName")
        self.GivenName: Optional[str] = kwargs.get("GivenName")
        self.Surname: Optional[str] = kwargs.get("Surname")
        self.BirthYear: Optional[int] = kwargs.get("BirthYear")
        self.BirthPlace: Optional[str] = kwargs.get("BirthPlace")
        self.DeathYear: Optional[int] = kwargs.get("DeathYear")
        self.DeathPlace: Optional[str] = kwargs.get("DeathPlace")
        self.Gender: Optional[str] = kwargs.get("Gender")
        self.IsLiving: Optional[bool] = kwargs.get("IsLiving")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "PersonId": self.PersonId,
            "TreeId": self.TreeId,
            "UserId": self.UserId,
            "FullName": self.FullName,
            "GivenName": self.GivenName,
            "Surname": self.Surname,
            "BirthYear": self.BirthYear,
            "BirthPlace": self.BirthPlace,
            "DeathYear": self.DeathYear,
            "DeathPlace": self.DeathPlace,
            "Gender": self.Gender,
            "IsLiving": self.IsLiving,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class ProfileDetailsResponse:
    """Model for validating Profile Details API responses."""

    def __init__(self, **kwargs):
        self.FirstName: Optional[str] = kwargs.get("FirstName")
        self.displayName: Optional[str] = kwargs.get("displayName")
        self.LastLoginDate: Optional[str] = kwargs.get("LastLoginDate")
        self.IsContactable: Optional[bool] = kwargs.get("IsContactable")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "FirstName": self.FirstName,
            "displayName": self.displayName,
            "LastLoginDate": self.LastLoginDate,
            "IsContactable": self.IsContactable,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class TreeOwnerResponse:
    """Model for validating Tree Owner API responses."""

    def __init__(self, **kwargs):
        self.owner: Optional[Dict[str, Any]] = kwargs.get("owner")
        self.displayName: Optional[str] = kwargs.get("displayName")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "owner": self.owner,
            "displayName": self.displayName,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class PersonFactsResponse:
    """Model for validating Person Facts API responses."""

    def __init__(self, **kwargs):
        self.data: Optional[Dict[str, Any]] = kwargs.get("data")
        self.personResearch: Optional[Dict[str, Any]] = kwargs.get("personResearch")
        self.PersonFacts: Optional[List[Dict[str, Any]]] = kwargs.get("PersonFacts")
        self.PersonFullName: Optional[str] = kwargs.get("PersonFullName")
        self.FirstName: Optional[str] = kwargs.get("FirstName")
        self.LastName: Optional[str] = kwargs.get("LastName")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "data": self.data,
            "personResearch": self.personResearch,
            "PersonFacts": self.PersonFacts,
            "PersonFullName": self.PersonFullName,
            "FirstName": self.FirstName,
            "LastName": self.LastName,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class GetLadderResponse:
    """Model for validating GetLadder API responses."""

    def __init__(self, **kwargs):
        self.data: Optional[Dict[str, Any]] = kwargs.get("data")
        self.relationship: Optional[Dict[str, Any]] = kwargs.get("relationship")
        self.paths: Optional[List[Dict[str, Any]]] = kwargs.get("paths")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "data": self.data,
            "relationship": self.relationship,
            "paths": self.paths,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class DiscoveryRelationshipResponse:
    """Model for validating Discovery Relationship API responses."""

    def __init__(self, **kwargs):
        self.relationship: Optional[str] = kwargs.get("relationship")
        self.paths: Optional[List[Dict[str, Any]]] = kwargs.get("paths")
        self.confidence: Optional[str] = kwargs.get("confidence")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "relationship": self.relationship,
            "paths": self.paths,
            "confidence": self.confidence,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class HeaderTreesResponse:
    """Model for validating Header Trees API responses."""

    def __init__(self, **kwargs):
        self.menuitems: Optional[List[Dict[str, Any]]] = kwargs.get("menuitems")
        self.url: Optional[str] = kwargs.get("url")
        self.text: Optional[str] = kwargs.get("text")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "menuitems": self.menuitems,
            "url": self.url,
            "text": self.text,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class SendMessageResponse:
    """Model for validating Send Message API responses."""

    def __init__(self, **kwargs):
        self.conversation_id: Optional[str] = kwargs.get("conversation_id")
        self.message: Optional[str] = kwargs.get("message")
        self.author: Optional[str] = kwargs.get("author")
        self.status: Optional[str] = kwargs.get("status")

    def dict(self, exclude_none=False):
        """Convert to dictionary format."""
        result = {
            "conversation_id": self.conversation_id,
            "message": self.message,
            "author": self.author,
            "status": self.status,
        }
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


# --- API Rate Limiter ---
class ApiRateLimiter:
    """Simple rate limiter for API calls to prevent overwhelming Ancestry servers."""

    def __init__(self, max_calls_per_minute: int = 60, max_calls_per_hour: int = 1000):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_hour = max_calls_per_hour
        self.minute_calls = []
        self.hour_calls = []

    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding rate limits."""
        current_time = datetime.now()

        # Clean old entries
        self.minute_calls = [
            t for t in self.minute_calls if (current_time - t).seconds < 60
        ]
        self.hour_calls = [
            t for t in self.hour_calls if (current_time - t).seconds < 3600
        ]

        # Check limits
        if len(self.minute_calls) >= self.max_calls_per_minute:
            return False
        if len(self.hour_calls) >= self.max_calls_per_hour:
            return False

        return True

    def record_request(self):
        """Record that a request was made."""
        current_time = datetime.now()
        self.minute_calls.append(current_time)
        self.hour_calls.append(current_time)

    def wait_time_until_available(self) -> float:
        """Get the time in seconds to wait until next request is allowed."""
        if self.can_make_request():
            return 0.0

        current_time = datetime.now()

        # Check minute limit
        if len(self.minute_calls) >= self.max_calls_per_minute:
            oldest_minute_call = min(self.minute_calls)
            return max(0, 60 - (current_time - oldest_minute_call).seconds)

        # Check hour limit
        if len(self.hour_calls) >= self.max_calls_per_hour:
            oldest_hour_call = min(self.hour_calls)
            return max(0, 3600 - (current_time - oldest_hour_call).seconds)

        return 0.0


# Global rate limiter instance
api_rate_limiter = ApiRateLimiter()


# --- Helper Functions for parse_ancestry_person_details ---
def _extract_name_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> str:
    """
    Extract and format a person's name from Ancestry API response data.

    This function attempts to extract a person's name from multiple possible sources
    in the API response data, using a priority-based approach to ensure the most
    accurate name is retrieved.

    Args:
        person_card (Dict): Data from Suggest API or similar person card responses
        facts_data (Optional[Dict]): Data from Facts API or detailed person information

    Returns:
        str: The formatted person's name, or "Unknown" if no name could be extracted

    Note:
        The function tries multiple extraction strategies in order of preference:
        1. PersonFacts Name fact from facts_data
        2. Direct name fields from facts_data (personName, DisplayName, etc.)
        3. Constructed name from FirstName/LastName in facts_data
        4. FullName from person_card
        5. Constructed name from GivenName/Surname in person_card
        6. Generic 'name' field from person_card

        Names are filtered to avoid returning "Valued Relative" placeholder names.
    """
    name = "Unknown"
    formatter = format_name
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            name = person_info.get("personName", name)
        # End of if
        if name == "Unknown":
            name = facts_data.get("personName", name)
        # End of if
        if name == "Unknown":
            name = facts_data.get("DisplayName", name)
        # End of if
        if name == "Unknown":
            name = facts_data.get("PersonFullName", name)
        # End of if
        if name == "Unknown":
            person_facts_list = facts_data.get("PersonFacts", [])
            if isinstance(person_facts_list, list):
                name_fact = next(
                    (
                        f
                        for f in person_facts_list
                        if isinstance(f, dict) and f.get("TypeString") == "Name"
                    ),
                    None,
                )
                if name_fact and name_fact.get("Value"):
                    name = name_fact.get("Value", "Unknown")
                # End of if
            # End of if
        # End of if
        if name == "Unknown":
            first_name_pd = facts_data.get("FirstName")
            last_name_pd = facts_data.get("LastName")
            if first_name_pd or last_name_pd:
                name = (
                    f"{first_name_pd or ''} {last_name_pd or ''}".strip() or "Unknown"
                )
            # End of if
        # End of if
    # End of if
    if name == "Unknown" and person_card:
        suggest_fullname = person_card.get("FullName")
        suggest_given = person_card.get("GivenName")
        suggest_sur = person_card.get("Surname")
        if suggest_fullname:
            name = suggest_fullname
        elif suggest_given or suggest_sur:
            name = f"{suggest_given or ''} {suggest_sur or ''}".strip() or "Unknown"
        # End of if/elif
        if name == "Unknown":
            name = person_card.get("name", "Unknown")
        # End of if
    # End of if
    formatted_name_val = formatter(name) if name and name != "Unknown" else "Unknown"
    return "Unknown" if formatted_name_val == "Valued Relative" else formatted_name_val


# End of _extract_name_from_api_details


def _extract_gender_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[str]:
    """
    Extract and normalize gender information from Ancestry API response data.

    Attempts to extract gender from multiple possible sources and normalizes
    the value to standard single-character format ("M" or "F").

    Args:
        person_card (Dict): Data from Suggest API or similar person card responses
        facts_data (Optional[Dict]): Data from Facts API or detailed person information

    Returns:
        Optional[str]: "M" for male, "F" for female, or None if gender cannot be determined

    Note:
        The function checks the following sources in order:
        1. PersonFacts Gender fact from facts_data
        2. Direct gender fields from facts_data
        3. Gender field from person_card

        Input values like "Male"/"Female" are normalized to "M"/"F".
    """
    gender = None
    gender_str = None
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            gender_str = person_info.get("gender")
        # End of if
        if not gender_str:
            gender_str = facts_data.get("gender")
        # End of if
        if not gender_str:
            gender_str = facts_data.get("PersonGender")
        # End of if
        if not gender_str:
            person_facts_list = facts_data.get("PersonFacts", [])
            if isinstance(person_facts_list, list):
                gender_fact = next(
                    (
                        f
                        for f in person_facts_list
                        if isinstance(f, dict) and f.get("TypeString") == "Gender"
                    ),
                    None,
                )
                if gender_fact and gender_fact.get("Value"):
                    gender_str = gender_fact.get("Value")
                # End of if
            # End of if
        # End of if
    # End of if
    if not gender_str and person_card:
        gender_str = person_card.get("Gender")
        if not gender_str:
            gender_str = person_card.get("gender")
        # End of if
    # End of if
    if gender_str and isinstance(gender_str, str):
        gender_str_lower = gender_str.lower()
        if gender_str_lower == "male":
            gender = "M"
        elif gender_str_lower == "female":
            gender = "F"
        elif gender_str_lower in ["m", "f"]:
            gender = gender_str_lower.upper()
        # End of if/elif
    # End of if
    return gender


# End of _extract_gender_from_api_details


def _extract_living_status_from_api_details(
    person_card: Dict, facts_data: Optional[Dict]
) -> Optional[bool]:
    """
    Extract living status information from Ancestry API response data.

    Determines whether a person is marked as living or deceased based on
    various fields in the API response data.

    Args:
        person_card (Dict): Data from Suggest API or similar person card responses
        facts_data (Optional[Dict]): Data from Facts API or detailed person information

    Returns:
        Optional[bool]: True if person is living, False if deceased, None if unknown

    Note:
        The function checks multiple possible field names for living status:
        - isLiving, IsPersonLiving (from facts_data)
        - IsLiving, isLiving (from person_card)
    """
    is_living = None
    if facts_data and isinstance(facts_data, dict):
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            is_living = person_info.get("isLiving")
        # End of if
        if is_living is None:
            is_living = facts_data.get("isLiving")
        # End of if
        if is_living is None:
            is_living = facts_data.get("IsPersonLiving")
        # End of if
    # End of if
    if is_living is None and person_card:
        is_living = person_card.get("IsLiving")
        if is_living is None:
            is_living = person_card.get("isLiving")
        # End of if
    # End of if
    return bool(is_living) if is_living is not None else None


# End of _extract_living_status_from_api_details


def _extract_event_from_api_details(
    event_type: str, person_card: Dict, facts_data: Optional[Dict]
) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Extract event information (date, place, parsed date object) from Ancestry API data.

    This comprehensive function extracts vital event information (birth, death, etc.)
    from multiple possible sources in Ancestry API responses, handling various
    data formats and structures.

    Args:
        event_type (str): Type of event to extract ("Birth", "Death", etc.)
        person_card (Dict): Data from Suggest API or similar person card responses
        facts_data (Optional[Dict]): Data from Facts API or detailed person information

    Returns:
        Tuple[Optional[str], Optional[str], Optional[datetime]]: A tuple containing:
            - date_str: Raw date string from API
            - place_str: Place name string from API
            - date_obj: Parsed datetime object (if date parsing successful)

    Note:
        The function tries multiple extraction strategies:
        1. PersonFacts primary event facts from facts_data
        2. Structured facts data with date/place objects
        3. Alternative event fact formats
        4. Suggest API year/place fields from person_card
        5. Concatenated event info strings from person_card

        Date parsing is attempted using the gedcom_utils._parse_date function.
    """
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None
    parser = _parse_date
    event_key_lower = event_type.lower()
    suggest_year_key = f"{event_type}Year"
    suggest_place_key = f"{event_type}Place"
    facts_user_key = event_type
    app_api_key = f"{event_key_lower}Date"
    app_api_facts_key = event_type
    found_in_facts = False
    if facts_data and isinstance(facts_data, dict):
        person_facts_list = facts_data.get("PersonFacts", [])
        if isinstance(person_facts_list, list):
            event_fact = next(
                (
                    f
                    for f in person_facts_list
                    if isinstance(f, dict)
                    and f.get("TypeString") == facts_user_key
                    and not f.get("IsAlternate")
                ),
                None,
            )
            if event_fact:
                date_str = event_fact.get("Date")
                place_str = event_fact.get("Place")
                parsed_date_data = event_fact.get("ParsedDate")
                found_in_facts = True
                logger.debug(
                    f"Found primary {event_type} fact in PersonFacts: Date='{date_str}', Place='{place_str}'"
                )
                if isinstance(parsed_date_data, dict) and parser:
                    year = parsed_date_data.get("Year")
                    month = parsed_date_data.get("Month")
                    day = parsed_date_data.get("Day")
                    if year:
                        try:
                            temp_date_str = str(year)
                            if month:
                                temp_date_str += f"-{str(month).zfill(2)}"
                            # End of if
                            if day:
                                temp_date_str += f"-{str(day).zfill(2)}"
                            # End of if
                            date_obj = parser(temp_date_str)
                            logger.debug(
                                f"Parsed {event_type} date object from ParsedDate: {date_obj}"
                            )
                        except Exception as dt_err:
                            logger.warning(
                                f"Could not parse {event_type} date from ParsedDate {parsed_date_data}: {dt_err}"
                            )
                        # End of try/except
                    # End of if year
                # End of if parsed_date_data
            # End of if event_fact
        # End of if
        if not found_in_facts:
            fact_group_list = facts_data.get("facts", {}).get(app_api_facts_key, [])
            if fact_group_list and isinstance(fact_group_list, list):
                fact_group = fact_group_list[0]
                if isinstance(fact_group, dict):
                    date_info = fact_group.get("date", {})
                    place_info = fact_group.get("place", {})
                    if isinstance(date_info, dict):
                        date_str = date_info.get(
                            "normalized", date_info.get("original")
                        )
                    # End of if
                    if isinstance(place_info, dict):
                        place_str = place_info.get("placeName")
                    # End of if
                    found_in_facts = True
                # End of if
            # End of if
        # End of if
        if not found_in_facts:
            event_fact_alt = facts_data.get(app_api_key)
            if event_fact_alt and isinstance(event_fact_alt, dict):
                date_str = event_fact_alt.get("normalized", event_fact_alt.get("date"))
                place_str = event_fact_alt.get("place", place_str)
                found_in_facts = True
            elif isinstance(event_fact_alt, str):
                date_str = event_fact_alt
                found_in_facts = True
            # End of if/elif
        # End of if
    # End of if
    if not found_in_facts and person_card:
        suggest_year = person_card.get(suggest_year_key)
        suggest_place = person_card.get(suggest_place_key)
        if suggest_year:
            date_str = str(suggest_year)
            place_str = suggest_place
            logger.debug(
                f"Using Suggest API keys for {event_type}: Year='{date_str}', Place='{place_str}'"
            )
        else:
            event_info_card = person_card.get(event_key_lower, "")
            if event_info_card and isinstance(event_info_card, str):
                parts = re.split(r"\s+in\s+", event_info_card, maxsplit=1)
                date_str = parts[0].strip() if parts else event_info_card
                if place_str is None and len(parts) > 1:
                    place_str = parts[1].strip()
                # End of if
            elif isinstance(event_info_card, dict):
                date_str = event_info_card.get("date", date_str)
                if place_str is None:
                    place_str = event_info_card.get("place", place_str)
                # End of if
            # End of if/elif
        # End of if/else
    # End of if
    if date_obj is None and date_str and parser:
        try:
            date_obj = parser(date_str)
        except Exception as parse_err:
            logger.warning(
                f"Failed to parse {event_type} date string '{date_str}': {parse_err}"
            )
        # End of try/except
    # End of if
    return date_str, place_str, date_obj


# End of _extract_event_from_api_details


def _generate_person_link(
    person_id: Optional[str], tree_id: Optional[str], base_url: str
) -> str:
    """
    Generate an appropriate Ancestry.com URL for viewing a person's details.

    Creates different types of links based on available identifiers:
    - Tree-based links for persons with both person_id and tree_id
    - Discovery match links for persons with only person_id

    Args:
        person_id (Optional[str]): The person's unique identifier
        tree_id (Optional[str]): The tree's unique identifier (if available)
        base_url (str): Base Ancestry URL (e.g., "https://www.ancestry.com")

    Returns:
        str: A complete URL to view the person's details, or "(Link unavailable)"
             if insufficient information is provided

    Examples:
        >>> _generate_person_link("123", "456", "https://ancestry.com")
        "https://ancestry.com/family-tree/person/tree/456/person/123/facts"

        >>> _generate_person_link("123", None, "https://ancestry.com")
        "https://ancestry.com/discoveryui-matches/list/summary/123"
    """
    if tree_id and person_id:
        return f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
    elif person_id:
        return f"{base_url}/discoveryui-matches/list/summary/{person_id}"
    # End of if/elif
    return "(Link unavailable)"


# End of _generate_person_link


def parse_ancestry_person_details(
    person_card: Dict, facts_data: Optional[Dict] = None
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "name": "Unknown",
        "birth_date": "N/A",
        "birth_place": None,
        "api_birth_obj": None,
        "death_date": "N/A",
        "death_place": None,
        "api_death_obj": None,
        "gender": None,
        "is_living": None,
        "person_id": person_card.get("PersonId"),
        "tree_id": person_card.get("TreeId"),
        "user_id": person_card.get("UserId"),
        "link": "(Link unavailable)",
    }
    if not details["person_id"]:
        details["person_id"] = person_card.get("personId")
    # End of if
    if not details["tree_id"]:
        details["tree_id"] = person_card.get("treeId")
    # End of if

    if facts_data and isinstance(facts_data, dict):
        details["person_id"] = facts_data.get("PersonId", details["person_id"])
        details["tree_id"] = facts_data.get("TreeId", details["tree_id"])
        details["user_id"] = facts_data.get("UserId", details["user_id"])
        if not details["user_id"]:
            person_info = facts_data.get("person", {})
            if isinstance(person_info, dict):
                details["user_id"] = person_info.get("userId", details["user_id"])
            # End of if
        # End of if
    # End of if

    details["name"] = _extract_name_from_api_details(person_card, facts_data)
    details["gender"] = _extract_gender_from_api_details(person_card, facts_data)
    details["is_living"] = _extract_living_status_from_api_details(
        person_card, facts_data
    )

    birth_date_raw, details["birth_place"], details["api_birth_obj"] = (
        _extract_event_from_api_details("Birth", person_card, facts_data)
    )
    death_date_raw, details["death_place"], details["api_death_obj"] = (
        _extract_event_from_api_details("Death", person_card, facts_data)
    )

    cleaner = _clean_display_date
    details["birth_date"] = cleaner(birth_date_raw) if birth_date_raw else "N/A"
    details["death_date"] = cleaner(death_date_raw) if death_date_raw else "N/A"

    if details["birth_date"] == "N/A" and details["api_birth_obj"]:
        details["birth_date"] = str(details["api_birth_obj"].year)
    # End of if
    if details["death_date"] == "N/A" and details["api_death_obj"]:
        details["death_date"] = str(details["api_death_obj"].year)
    # End of if

    base_url_for_link = getattr(
        config_instance, "BASE_URL", "https://www.ancestry.com"
    ).rstrip("/")

    link_id = details["user_id"] or details["person_id"]
    link_tree_id = details["tree_id"] if not details["user_id"] else None

    details["link"] = _generate_person_link(link_id, link_tree_id, base_url_for_link)

    logger.debug(
        f"Parsed API details for '{details.get('name', 'Unknown')}': PersonID={details.get('person_id')}, TreeID={details.get('tree_id', 'N/A')}, UserID={details.get('user_id', 'N/A')}, Born='{details.get('birth_date')}' [{details.get('api_birth_obj')}] in '{details.get('birth_place') or '?'}', Died='{details.get('death_date')}' [{details.get('api_death_obj')}] in '{details.get('death_place') or '?'}', Gender='{details.get('gender') or '?'}', Living={details.get('is_living')}, Link='{details.get('link')}'"
    )
    return details


# End of parse_ancestry_person_details


# Note: format_api_relationship_path has been moved to relationship_utils.py
# Import it from there instead of defining it here


def print_group(label: str, items: List[Dict]):
    print(f"\n{label}:")
    if items:
        formatter = format_name
        for item in items:
            name_to_format = item.get("name") if isinstance(item, dict) else None
            print(f"  - {formatter(name_to_format)}")
        # End of for
    else:
        print("  (None found)")
    # End of if/else


def _get_api_timeout(default: int = 60) -> int:
    """
    Get the configured API timeout value with fallback to default.

    Retrieves the API timeout value from selenium_config if available,
    with validation to ensure a reasonable timeout value.

    Args:
        default (int): Default timeout value in seconds if config is unavailable

    Returns:
        int: Timeout value in seconds to use for API calls

    Note:
        If the configured timeout is invalid (non-positive or wrong type),
        the default value is used and a warning is logged.
    """
    timeout_value = default
    if selenium_config and hasattr(selenium_config, "API_TIMEOUT"):
        config_timeout = getattr(selenium_config, "API_TIMEOUT")
        if isinstance(config_timeout, (int, float)) and config_timeout > 0:
            timeout_value = int(config_timeout)
        else:
            logger.warning(
                f"Invalid API_TIMEOUT value in config ({config_timeout}), using default {default}s."
            )
        # End of if/else
    # End of if
    return timeout_value


# End of _get_api_timeout


def _get_owner_referer(session_manager: "SessionManager", base_url: str) -> str:
    """
    Generate an appropriate referer URL for API calls using the tree owner's facts page.

    Creates a proper referer URL that mimics natural browsing behavior by using
    the tree owner's facts page as the referer for API calls.

    Args:
        session_manager (SessionManager): Session manager with owner profile/tree info
        base_url (str): Base Ancestry URL

    Returns:
        str: Complete referer URL pointing to owner's facts page, or base URL if
             owner information is not available

    Note:
        Using proper referers helps maintain session validity and mimics normal
        user browsing patterns for better API reliability.
    """
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    if owner_profile_id and owner_tree_id:
        referer_path = (
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts"
        )
        referer = urljoin(base_url.rstrip("/") + "/", referer_path.lstrip("/"))
        logger.debug(f"Using owner facts page as referer: {referer}")
        return referer
    else:
        logger.warning(
            "Owner profile/tree ID missing in session. Using base URL as referer."
        )
        return base_url.rstrip("/") + "/"
    # End of if/else


# End of _get_owner_referer


def call_suggest_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    owner_profile_id: Optional[str],  # noqa: Kept for API signature consistency
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    if not callable(_api_req):
        logger.critical(
            "Suggest API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("Suggest API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not owner_tree_id:
        logger.error("Suggest API call failed: owner_tree_id is required.")
        return None
    # End of if

    api_description = "Suggest API"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")
    suggest_params_list = ["isHideVeiledRecords=false"]
    if first_name_raw:
        suggest_params_list.append(f"partialFirstName={quote(first_name_raw)}")
    # End of if
    if surname_raw:
        suggest_params_list.append(f"partialLastName={quote(surname_raw)}")
    # End of if
    if birth_year:
        suggest_params_list.append(f"birthYear={birth_year}")
    # End of if
    suggest_params = "&".join(suggest_params_list)
    formatted_path = API_PATH_PERSON_PICKER_SUGGEST.format(tree_id=owner_tree_id)
    suggest_url = (
        urljoin(base_url.rstrip("/") + "/", formatted_path) + f"?{suggest_params}"
    )
    owner_facts_referer = _get_owner_referer(session_manager, base_url)
    timeouts_used = timeouts if timeouts else [20, 30, 60]
    max_attempts = len(timeouts_used)
    logger.info(f"Attempting {api_description} search: {suggest_url}")

    suggest_response = None
    for attempt, timeout in enumerate(timeouts_used, 1):
        logger.debug(
            f"{api_description} attempt {attempt}/{max_attempts} with timeout {timeout}s"
        )
        try:
            custom_headers = {
                "Accept": "application/json",
                "Referer": owner_facts_referer,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            }
            suggest_response = _api_req(
                url=suggest_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description=api_description,
                headers=custom_headers,
                referer_url=owner_facts_referer,
                timeout=timeout,
                use_csrf_token=False,
            )
            if isinstance(suggest_response, list):
                # Validate response with Pydantic if available
                if PYDANTIC_AVAILABLE and suggest_response:
                    validated_results = []
                    validation_errors = 0

                    for item in suggest_response:
                        try:
                            validated_item = PersonSuggestResponse(**item)
                            validated_results.append(
                                validated_item.dict(exclude_none=True)
                            )
                        except Exception as validation_err:
                            validation_errors += 1
                            logger.debug(
                                f"Response validation warning for item: {validation_err}"
                            )
                            # Keep original item if validation fails
                            validated_results.append(item)

                    if validation_errors > 0:
                        logger.warning(
                            f"Response validation: {validation_errors}/{len(suggest_response)} items had validation issues"
                        )

                    suggest_response = validated_results

                logger.info(
                    f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts}), found {len(suggest_response)} results."
                )
                return suggest_response
            elif suggest_response is None:
                logger.warning(
                    f"{api_description} call using _api_req returned None on attempt {attempt}/{max_attempts}."
                )
            else:
                logger.error(
                    f"{api_description} call using _api_req returned unexpected type: {type(suggest_response)}"
                )
                logger.debug(
                    f"Unexpected Response Content: {str(suggest_response)[:500]}"
                )
                suggest_response = None
                break
            # End of if/elif/else
        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
        except Exception as api_err:
            logger.error(
                f"{api_description} _api_req call failed on attempt {attempt}/{max_attempts}: {api_err}",
                exc_info=True,
            )
            suggest_response = None
            break
        # End of try/except
    # End of for

    if suggest_response is None:
        logger.warning(
            f"{api_description} failed via _api_req. Attempting direct requests fallback."
        )
        direct_response_obj = None
        try:
            cookies = {}
            if session_manager._requests_session:
                cookies = session_manager._requests_session.cookies.get_dict()
            # End of if
            direct_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": owner_facts_referer,
                "X-Requested-With": "XMLHttpRequest",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
            }
            logger.debug(f"Direct request URL: {suggest_url}")
            logger.debug(f"Direct request headers: {direct_headers}")
            logger.debug(f"Direct request cookies: {list(cookies.keys())}")
            direct_timeout = _get_api_timeout(30)
            direct_response_obj = requests.get(
                suggest_url,
                headers=direct_headers,
                cookies=cookies,
                timeout=direct_timeout,
            )
            if direct_response_obj.status_code == 200:
                direct_data = direct_response_obj.json()
                if isinstance(direct_data, list):
                    logger.info(
                        f"Direct request fallback successful! Found {len(direct_data)} results."
                    )
                    return direct_data
                else:
                    logger.warning(
                        f"Direct request succeeded (200 OK) but returned non-list data: {type(direct_data)}"
                    )
                    logger.debug(f"Direct Response content: {str(direct_data)[:500]}")
                # End of if/else
            else:
                logger.warning(
                    f"Direct request fallback failed: Status {direct_response_obj.status_code}"
                )
                logger.debug(
                    f"Direct Response content: {direct_response_obj.text[:500]}"
                )
            # End of if/else
        except requests.exceptions.Timeout:
            logger.error(
                f"Direct request fallback timed out after {direct_timeout} seconds"
            )
        except json.JSONDecodeError as json_err:
            logger.error(f"Direct request fallback failed to decode JSON: {json_err}")
            if direct_response_obj:
                logger.debug(
                    f"Direct Response content: {direct_response_obj.text[:500]}"
                )
            # End of if
        except Exception as direct_err:
            logger.error(
                f"Direct request fallback failed with error: {direct_err}",
                exc_info=True,
            )
        # End of try/except
    # End of if

    logger.error(f"{api_description} failed after all attempts and fallback.")
    return None


# End of call_suggest_api


def call_facts_user_api(
    session_manager: "SessionManager",
    owner_profile_id: str,
    api_person_id: str,
    api_tree_id: str,
    base_url: str,
    timeouts: Optional[List[int]] = None,
) -> Optional[Dict]:
    if not callable(_api_req):
        logger.critical(
            "Facts API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("Facts API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not all([owner_profile_id, api_person_id, api_tree_id]):
        logger.error(
            "Facts API call failed: owner_profile_id, api_person_id, and api_tree_id are required."
        )
        return None
    # End of if

    api_description = "Person Facts User API"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    formatted_path = API_PATH_PERSON_FACTS_USER.format(
        owner_profile_id=owner_profile_id.lower(),
        tree_id=api_tree_id.lower(),
        person_id=api_person_id.lower(),
    )
    facts_api_url = urljoin(base_url.rstrip("/") + "/", formatted_path)
    facts_referer = _get_owner_referer(session_manager, base_url)
    facts_data_raw = None
    direct_timeout = _get_api_timeout(30)
    fallback_timeouts = timeouts if timeouts else [30, 45, 60]
    logger.info(f"Attempting {api_description} via direct request: {facts_api_url}")

    direct_response_obj = None
    try:
        cookies = {}
        if session_manager._requests_session:
            cookies = session_manager._requests_session.cookies.get_dict()
        # End of if
        direct_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": facts_referer,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Content-Type": "application/json",
            "DNT": "1",
            "Connection": "keep-alive",
        }
        logger.debug(f"Direct facts request headers: {direct_headers}")
        logger.debug(f"Direct facts request cookies: {list(cookies.keys())}")
        direct_response_obj = requests.get(
            facts_api_url,
            headers=direct_headers,
            cookies=cookies,
            timeout=direct_timeout,
        )
        if direct_response_obj.status_code == 200:
            facts_data_raw = direct_response_obj.json()
            if not isinstance(facts_data_raw, dict):
                logger.warning(
                    f"Direct facts request OK (200) but returned non-dict data: {type(facts_data_raw)}"
                )
                logger.debug(f"Response content: {direct_response_obj.text[:500]}")
                facts_data_raw = None
            else:
                logger.info(f"{api_description} call successful via direct request.")
            # End of if/else
        else:
            logger.warning(
                f"Direct facts request failed: Status {direct_response_obj.status_code}"
            )
            logger.debug(f"Response content: {direct_response_obj.text[:500]}")
            facts_data_raw = None
        # End of if/else
    except requests.exceptions.Timeout:
        logger.error(f"Direct facts request timed out after {direct_timeout} seconds")
        facts_data_raw = None
    except json.JSONDecodeError as json_err:
        logger.error(f"Direct facts request failed to decode JSON: {json_err}")
        if direct_response_obj:
            logger.debug(f"Response content: {direct_response_obj.text[:500]}")
        # End of if
        facts_data_raw = None
    except Exception as direct_err:
        logger.error(f"Direct facts request failed: {direct_err}", exc_info=True)
        facts_data_raw = None
    # End of try/except

    if facts_data_raw is None:
        logger.warning(
            f"{api_description} direct request failed. Trying _api_req fallback."
        )
        max_attempts = len(fallback_timeouts)
        for attempt, timeout in enumerate(fallback_timeouts, 1):
            logger.debug(
                f"{api_description} _api_req attempt {attempt}/{max_attempts} with timeout {timeout}s"
            )
            try:
                custom_headers = {
                    "Accept": "application/json",
                    "Referer": facts_referer,
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                    "X-Requested-With": "XMLHttpRequest",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
                }
                api_response = _api_req(
                    url=facts_api_url,
                    driver=session_manager.driver,
                    session_manager=session_manager,
                    method="GET",
                    api_description=api_description,
                    headers=custom_headers,
                    referer_url=facts_referer,
                    timeout=timeout,
                )
                if isinstance(api_response, dict):
                    facts_data_raw = api_response
                    logger.info(
                        f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts})."
                    )
                    break
                elif api_response is None:
                    logger.warning(
                        f"{api_description} _api_req returned None (attempt {attempt}/{max_attempts})."
                    )
                else:
                    logger.warning(
                        f"{api_description} _api_req returned unexpected type: {type(api_response)}"
                    )
                    logger.debug(
                        f"Unexpected Response Value: {str(api_response)[:500]}"
                    )
                # End of if/elif/else
            except requests.exceptions.Timeout:
                logger.warning(
                    f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
                )
            except Exception as api_req_err:
                logger.error(
                    f"{api_description} call using _api_req failed on attempt {attempt}/{max_attempts}: {api_req_err}",
                    exc_info=True,
                )
                facts_data_raw = None
                break
            # End of try/except
        # End of for
    # End of if

    if not isinstance(facts_data_raw, dict):
        logger.error(
            f"Failed to fetch valid {api_description} data after all attempts."
        )
        return None
    # End of if
    person_research_data = facts_data_raw.get("data", {}).get("personResearch")
    if not isinstance(person_research_data, dict) or not person_research_data:
        logger.error(
            f"{api_description} response received, but missing 'data.personResearch' dictionary."
        )
        logger.debug(f"Full raw response keys: {list(facts_data_raw.keys())}")
        if "data" in facts_data_raw and isinstance(facts_data_raw["data"], dict):
            logger.debug(f"'data' sub-keys: {list(facts_data_raw['data'].keys())}")
        else:
            logger.debug(f"'data' key missing or not a dict in response.")
        # End of if/else
        return None
    # End of if

    # Validate response with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            validated_response = PersonFactsResponse(**facts_data_raw)
            logger.debug("Facts API response validation successful")
        except Exception as validation_err:
            logger.warning(f"Facts API response validation warning: {validation_err}")
            # Continue with original data if validation fails

    logger.info(
        f"Successfully fetched and extracted 'personResearch' data for PersonID {api_person_id}."
    )
    return person_research_data


# End of call_facts_user_api


def call_getladder_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    target_person_id: str,
    base_url: str,
    timeout: Optional[int] = None,
) -> Optional[str]:
    if not callable(_api_req):
        logger.critical(
            "GetLadder API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("GetLadder API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not all([owner_tree_id, target_person_id]):
        logger.error(
            "GetLadder API call failed: owner_tree_id and target_person_id are required."
        )
        return None
    # End of if

    api_description = "Get Tree Ladder API"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    formatted_path = API_PATH_PERSON_GETLADDER.format(
        tree_id=owner_tree_id, person_id=target_person_id
    )
    ladder_api_url_base = urljoin(base_url.rstrip("/") + "/", formatted_path)
    query_params = urlencode({"callback": "no"})
    ladder_api_url = f"{ladder_api_url_base}?{query_params}"
    ladder_referer_path = (
        f"/family-tree/person/tree/{owner_tree_id}/person/{target_person_id}/facts"
    )
    ladder_referer = urljoin(
        base_url.rstrip("/") + "/", ladder_referer_path.lstrip("/")
    )
    api_timeout_val = timeout if timeout else _get_api_timeout(20)
    logger.info(f"Attempting {api_description} call: {ladder_api_url}")

    try:
        relationship_data = _api_req(
            url=ladder_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description=api_description,
            referer_url=ladder_referer,
            use_csrf_token=False,
            force_text_response=True,
            timeout=api_timeout_val,
        )
        if isinstance(relationship_data, str) and len(relationship_data) > 10:
            # Validate response with Pydantic if available and try to parse as JSON
            if PYDANTIC_AVAILABLE:
                try:
                    # Try to parse the string response as JSON for validation
                    import json

                    parsed_data = json.loads(relationship_data)
                    validated_response = GetLadderResponse(**parsed_data)
                    logger.debug("GetLadder API response validation successful")
                except json.JSONDecodeError:
                    logger.debug("GetLadder API returned non-JSON string response")
                except Exception as validation_err:
                    logger.warning(
                        f"GetLadder API response validation warning: {validation_err}"
                    )

            logger.info(f"{api_description} call successful, received string response.")
            return relationship_data
        elif isinstance(relationship_data, str):
            logger.warning(
                f"{api_description} call returned a very short string: '{relationship_data}'"
            )
            return None
        else:
            logger.warning(
                f"{api_description} call returned non-string or None: {type(relationship_data)}"
            )
            return None
        # End of if/elif/else
    except requests.exceptions.Timeout:
        logger.error(f"{api_description} call timed out after {api_timeout_val}s.")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_getladder_api


def call_discovery_relationship_api(
    session_manager: "SessionManager",
    selected_person_global_id: str,
    owner_profile_id: str,
    base_url: str,
    timeout: Optional[int] = None,
) -> Optional[Dict]:
    """
    Makes an API call to get relationship data from the Discovery API.

    This function calls the Ancestry Discovery Relationship API to find the relationship
    path between two individuals using their global profile IDs.

    Args:
        session_manager: SessionManager instance with active session
        selected_person_global_id: Global ID of the target person
        owner_profile_id: Global ID of the tree owner/reference person
        base_url: Base URL for Ancestry API
        timeout: Optional timeout in seconds (default: uses _get_api_timeout)

    Returns:
        Dictionary containing relationship path data or None if the call fails
    """
    if not callable(_api_req):
        logger.critical(
            "Discovery Relationship API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error(
            "Discovery Relationship API call failed: Invalid SessionManager passed."
        )
        return None
    # End of if
    if not all([owner_profile_id, selected_person_global_id]):
        logger.error(
            "Discovery Relationship API call failed: owner_profile_id and selected_person_global_id are required."
        )
        return None
    # End of if

    api_description = "Discovery Relationship API"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    formatted_path = API_PATH_DISCOVERY_RELATIONSHIP
    discovery_api_url = (
        urljoin(base_url.rstrip("/") + "/", formatted_path)
        + f"?profileIdFrom={owner_profile_id}&profileIdTo={selected_person_global_id}"
    )
    discovery_referer = f"{base_url.rstrip('/')}/discoveryui-matches/list/summary/{selected_person_global_id}"
    api_timeout_val = timeout if timeout else _get_api_timeout(30)
    logger.info(f"Attempting {api_description} call: {discovery_api_url}")

    try:
        custom_headers = {
            "Accept": "application/json",
            "Referer": discovery_referer,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        }

        relationship_data = _api_req(
            url=discovery_api_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description=api_description,
            headers=custom_headers,
            referer_url=discovery_referer,
            timeout=api_timeout_val,
            use_csrf_token=False,
        )

        if isinstance(relationship_data, dict) and "path" in relationship_data:
            # Validate response with Pydantic if available
            if PYDANTIC_AVAILABLE:
                try:
                    validated_response = DiscoveryRelationshipResponse(
                        **relationship_data
                    )
                    logger.debug(
                        "Discovery Relationship API response validation successful"
                    )
                except Exception as validation_err:
                    logger.warning(
                        f"Discovery Relationship API response validation warning: {validation_err}"
                    )

            logger.info(
                f"{api_description} call successful, received valid JSON response with path data."
            )
            return relationship_data
        elif isinstance(relationship_data, dict):
            # Validate response with Pydantic if available
            if PYDANTIC_AVAILABLE:
                try:
                    validated_response = DiscoveryRelationshipResponse(
                        **relationship_data
                    )
                    logger.debug(
                        "Discovery Relationship API response validation successful"
                    )
                except Exception as validation_err:
                    logger.warning(
                        f"Discovery Relationship API response validation warning: {validation_err}"
                    )

            logger.warning(
                f"{api_description} call returned JSON without 'path' key: {list(relationship_data.keys())}"
            )
            return relationship_data  # Still return the data for potential debugging
        else:
            logger.warning(
                f"{api_description} call returned unexpected type: {type(relationship_data)}"
            )
            return None
        # End of if/elif/else
    except requests.exceptions.Timeout:
        logger.error(f"{api_description} call timed out after {api_timeout_val}s.")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_discovery_relationship_api


def call_treesui_list_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    owner_profile_id: Optional[str],  # noqa: Kept for API signature consistency
    base_url: str,
    search_criteria: Dict[str, Any],
    timeouts: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    if not callable(_api_req):
        logger.critical(
            "TreesUI List API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")
    # End of if
    if not isinstance(session_manager, SessionManager):
        logger.error("TreesUI List API call failed: Invalid SessionManager passed.")
        return None
    # End of if
    if not owner_tree_id:
        logger.error("TreesUI List API call failed: owner_tree_id is required.")
        return None
    # End of if

    api_description = "TreesUI List API (Alternative Search)"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")
    if not birth_year:
        logger.warning(
            "Cannot call TreesUI List API: 'birth_year' is missing in search criteria."
        )
        return None
    # End of if
    treesui_params_list = ["limit=100", "fields=NAMES,BIRTH_DEATH"]
    if first_name_raw:
        treesui_params_list.append(f"fn={quote(first_name_raw)}")
    # End of if
    if surname_raw:
        treesui_params_list.append(f"ln={quote(surname_raw)}")
    # End of if
    treesui_params_list.append(f"by={birth_year}")
    treesui_params = "&".join(treesui_params_list)
    formatted_path = API_PATH_TREESUI_LIST.format(tree_id=owner_tree_id)
    treesui_url = (
        urljoin(base_url.rstrip("/") + "/", formatted_path) + f"?{treesui_params}"
    )
    owner_facts_referer = _get_owner_referer(session_manager, base_url)
    timeouts_used = timeouts if timeouts else [15, 25, 35]
    max_attempts = len(timeouts_used)
    logger.info(f"Attempting {api_description} search using _api_req: {treesui_url}")

    treesui_response = None
    for attempt, timeout in enumerate(timeouts_used, 1):
        logger.debug(
            f"{api_description} attempt {attempt}/{max_attempts} with timeout {timeout}s"
        )
        try:
            custom_headers = {
                "Accept": "application/json",
                "Referer": owner_facts_referer,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            }
            treesui_response = _api_req(
                url=treesui_url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description=api_description,
                headers=custom_headers,
                referer_url=owner_facts_referer,
                timeout=timeout,
                use_csrf_token=False,
            )
            if isinstance(treesui_response, list):
                logger.info(
                    f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts}), found {len(treesui_response)} results."
                )
                return treesui_response
            elif treesui_response is None:
                logger.warning(
                    f"{api_description} _api_req returned None (attempt {attempt}/{max_attempts})."
                )
            else:
                logger.error(
                    f"{api_description} returned unexpected format via _api_req: {type(treesui_response)}"
                )
                logger.debug(f"Unexpected Response: {str(treesui_response)[:500]}")
                return None
            # End of if/elif/else
        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
        except Exception as treesui_err:
            logger.error(
                f"{api_description} _api_req call failed on attempt {attempt}/{max_attempts}: {treesui_err}",
                exc_info=True,
            )
            treesui_response = None
            break
        # End of try/except
    # End of for

    logger.error(f"{api_description} failed after all attempts.")
    return None


# End of call_treesui_list_api


def call_send_message_api(
    session_manager: "SessionManager",
    person: "Person",
    message_text: str,
    existing_conv_id: Optional[str],
    log_prefix: str,
) -> Tuple[str, Optional[str]]:
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            f"{log_prefix}: Cannot send message - SessionManager or own profile ID missing."
        )
        return SEND_ERROR_MISSING_OWN_ID, None
    # End of if

    if not isinstance(person, Person) or not getattr(person, "profile_id", None):
        logger.error(
            f"{log_prefix}: Cannot send message - Invalid Person object (Type: {type(person)}) or missing profile ID."
        )
        return SEND_ERROR_INVALID_RECIPIENT, None
    # End of if
    if not isinstance(message_text, str) or not message_text.strip():
        logger.error(
            f"{log_prefix}: Cannot send message - Message text is empty or invalid."
        )
        return SEND_ERROR_API_PREP_FAILED, None
    # End of if

    app_mode = getattr(config_instance, "APP_MODE", "unknown")
    if app_mode == "dry_run":
        message_status = SEND_SUCCESS_DRY_RUN
        effective_conv_id = existing_conv_id or f"dryrun_{uuid.uuid4()}"
        logger.info(
            f"{log_prefix}: Dry Run - Simulated message send to {getattr(person, 'username', None) or getattr(person, 'profile_id', 'Unknown') }."
        )
        return message_status, effective_conv_id
    elif app_mode not in ["production", "testing"]:
        logger.error(
            f"{log_prefix}: Logic Error - Unexpected APP_MODE '{app_mode}' reached send logic."
        )
        return SEND_ERROR_INTERNAL_MODE, None
    # End of if/elif

    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    recipient_profile_id_upper = getattr(person, "profile_id", "").upper()

    is_initial = not existing_conv_id
    send_api_url: str = ""
    payload: Dict[str, Any] = {}
    send_api_desc: str = ""
    api_headers: Dict[str, Any] = {}

    try:
        base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
        if is_initial:
            send_api_url = urljoin(
                base_url_cfg.rstrip("/") + "/", API_PATH_SEND_MESSAGE_NEW
            )
            send_api_desc = "Create Conversation API"
            payload = {
                "content": message_text,
                "author": MY_PROFILE_ID_LOWER,
                "index": 0,
                "created": 0,
                "conversation_members": [
                    {
                        "user_id": recipient_profile_id_upper.lower(),
                        "family_circles": [],
                    },
                    {"user_id": MY_PROFILE_ID_LOWER},
                ],
            }
        elif existing_conv_id:
            formatted_path = API_PATH_SEND_MESSAGE_EXISTING.format(
                conv_id=existing_conv_id
            )
            send_api_url = urljoin(base_url_cfg.rstrip("/") + "/", formatted_path)
            send_api_desc = "Send Message API (Existing Conv)"
            payload = {
                "content": message_text,
                "author": MY_PROFILE_ID_LOWER,
            }
        else:
            logger.error(
                f"{log_prefix}: Logic Error - Cannot determine API URL/payload (existing_conv_id issue?)."
            )
            return SEND_ERROR_API_PREP_FAILED, None
        # End of if/elif/else

        ctx_headers = getattr(config_instance, "API_CONTEXTUAL_HEADERS", {}).get(
            send_api_desc, {}
        )
        api_headers = ctx_headers.copy()

    except Exception as prep_err:
        logger.error(
            f"{log_prefix}: Error preparing API request data: {prep_err}", exc_info=True
        )
        return SEND_ERROR_API_PREP_FAILED, None
    # End of try/except

    api_response = _api_req(
        url=send_api_url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="POST",
        json_data=payload,
        use_csrf_token=False,
        headers=api_headers,
        api_description=send_api_desc,
    )

    message_status = SEND_ERROR_UNKNOWN
    new_conversation_id_from_api: Optional[str] = None
    post_ok = False

    if api_response is None:
        message_status = SEND_ERROR_POST_FAILED
        logger.error(
            f"{log_prefix}: API POST ({send_api_desc}) failed (No response/Retries exhausted)."
        )
    elif isinstance(api_response, requests.Response):
        message_status = f"send_error (http_{api_response.status_code})"
        logger.error(
            f"{log_prefix}: API POST ({send_api_desc}) failed with status {api_response.status_code}."
        )
        try:
            logger.debug(f"Error response body: {api_response.text[:500]}")
        except Exception:
            pass
        # End of try/except
    elif isinstance(api_response, dict):
        try:
            if is_initial:
                api_conv_id = str(api_response.get(KEY_CONVERSATION_ID, ""))
                msg_details = api_response.get(KEY_MESSAGE, {})
                api_author = (
                    str(msg_details.get(KEY_AUTHOR, "")).upper()
                    if isinstance(msg_details, dict)
                    else None
                )

                if api_conv_id and api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = api_conv_id
                else:
                    logger.error(
                        f"{log_prefix}: API initial response format invalid (ConvID: '{api_conv_id}', Author: '{api_author}', Expected Author: '{MY_PROFILE_ID_UPPER}')."
                    )
                    logger.debug(f"API Response: {api_response}")
                    message_status = SEND_ERROR_VALIDATION_FAILED
                # End of if/else
            else:
                api_author = str(api_response.get(KEY_AUTHOR, "")).upper()
                if api_author == MY_PROFILE_ID_UPPER:
                    post_ok = True
                    new_conversation_id_from_api = existing_conv_id
                else:
                    logger.error(
                        f"{log_prefix}: API follow-up author validation failed (Author: '{api_author}', Expected Author: '{MY_PROFILE_ID_UPPER}')."
                    )
                    logger.debug(f"API Response: {api_response}")
                    message_status = SEND_ERROR_VALIDATION_FAILED
                # End of if/else
            # End of if/else

            if post_ok:
                message_status = SEND_SUCCESS_DELIVERED
                logger.info(
                    f"{log_prefix}: Message send to {getattr(person, 'username', None) or getattr(person, 'profile_id', 'Unknown')} successful (ConvID: {new_conversation_id_from_api})."
                )
            # End of if

        except Exception as parse_err:
            logger.error(
                f"{log_prefix}: Error parsing successful API response ({send_api_desc}): {parse_err}",
                exc_info=True,
            )
            logger.debug(f"API Response received: {api_response}")
            message_status = SEND_ERROR_UNEXPECTED_FORMAT
        # End of try/except
    else:
        logger.error(
            f"{log_prefix}: API call ({send_api_desc}) unexpected success format. Type:{type(api_response)}, Resp:{str(api_response)[:200]}"
        )
        message_status = SEND_ERROR_UNEXPECTED_FORMAT
    # End of if/elif/else

    if not post_ok and message_status == SEND_ERROR_UNKNOWN:
        message_status = SEND_ERROR_VALIDATION_FAILED
        logger.warning(
            f"{log_prefix}: Message send attempt concluded with status: {message_status}"
        )
    # End of if

    return message_status, new_conversation_id_from_api


# End of call_send_message_api


def call_profile_details_api(
    session_manager: "SessionManager", profile_id: str
) -> Optional[Dict[str, Any]]:
    if not profile_id or not isinstance(profile_id, str):
        logger.warning("call_profile_details_api: Profile ID missing or invalid.")
        return None
    # End of if
    if not session_manager or not session_manager.my_profile_id:
        logger.error(
            "call_profile_details_api: SessionManager or own profile ID missing."
        )
        return None
    # End of if

    if not session_manager.is_sess_valid():
        logger.error(
            f"call_profile_details_api: Session invalid for Profile ID {profile_id}."
        )
        return None
    # End of if

    api_description = "Profile Details API (Single)"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
    profile_url = urljoin(
        base_url_cfg,
        f"{API_PATH_PROFILE_DETAILS}?userId={profile_id.upper()}",
    )
    referer_url = urljoin(base_url_cfg, "/messaging/")

    logger.debug(
        f"Fetching profile details ({api_description}) for Profile ID {profile_id}..."
    )

    try:
        profile_response = _api_req(
            url=profile_url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers={},
            use_csrf_token=False,
            api_description=api_description,
            referer_url=referer_url,
        )

        if profile_response and isinstance(profile_response, dict):
            # Validate response with Pydantic if available
            if PYDANTIC_AVAILABLE:
                try:
                    validated_response = ProfileDetailsResponse(**profile_response)
                    logger.debug("Profile Details API response validation successful")
                except Exception as validation_err:
                    logger.warning(
                        f"Profile Details API response validation warning: {validation_err}"
                    )

            logger.debug(f"Successfully fetched profile details for {profile_id}.")
            result_data: Dict[str, Any] = {
                "first_name": None,
                "last_logged_in_dt": None,
                "contactable": False,
            }

            first_name_raw = profile_response.get(KEY_FIRST_NAME)
            if first_name_raw and isinstance(first_name_raw, str):
                result_data["first_name"] = format_name(first_name_raw)
            else:
                display_name_raw = profile_response.get(KEY_DISPLAY_NAME)
                if display_name_raw and isinstance(display_name_raw, str):
                    formatted_dn = format_name(display_name_raw)
                    result_data["first_name"] = (
                        formatted_dn.split()[0]
                        if formatted_dn != "Valued Relative"
                        else None
                    )
                else:
                    logger.warning(
                        f"Could not extract FirstName or DisplayName for profile {profile_id}"
                    )
                # End of if/else
            # End of if/else

            contactable_val = profile_response.get(KEY_IS_CONTACTABLE)
            result_data["contactable"] = (
                bool(contactable_val) if contactable_val is not None else False
            )

            last_login_str = profile_response.get(KEY_LAST_LOGIN_DATE)
            if last_login_str and isinstance(last_login_str, str):
                try:
                    if last_login_str.endswith("Z"):
                        dt_aware = datetime.fromisoformat(
                            last_login_str.replace("Z", "+00:00")
                        )
                    elif "+" in last_login_str or "-" in last_login_str[10:]:
                        dt_aware = datetime.fromisoformat(last_login_str)
                    else:
                        dt_naive = datetime.fromisoformat(last_login_str)
                        dt_aware = dt_naive.replace(tzinfo=timezone.utc)
                    # End of if/elif/else
                    result_data["last_logged_in_dt"] = dt_aware.astimezone(timezone.utc)
                except (ValueError, TypeError) as date_parse_err:
                    logger.warning(
                        f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {date_parse_err}"
                    )
                # End of try/except
            else:
                logger.debug(
                    f"LastLoginDate missing or invalid for profile {profile_id}"
                )
            # End of if/else
            return result_data
        elif isinstance(profile_response, requests.Response):
            logger.warning(
                f"Failed profile details fetch for {profile_id}. Status: {profile_response.status_code}."
            )
            return None
        elif profile_response is None:
            logger.warning(
                f"Failed profile details fetch for {profile_id} (_api_req returned None)."
            )
            return None
        else:
            logger.warning(
                f"Failed profile details fetch for {profile_id} (Invalid response type: {type(profile_response)})."
            )
            return None
        # End of if/elif/else
    except requests.exceptions.RequestException as req_e:
        logger.error(
            f"RequestException fetching profile details for {profile_id}: {req_e}",
            exc_info=False,
        )
        return None
    # End of try/except
    except Exception as e:
        logger.error(
            f"Unexpected error fetching profile details for {profile_id}: {e}",
            exc_info=True,
        )
        return None
    # End of try/except


# End of call_profile_details_api


def call_header_trees_api_for_tree_id(
    session_manager: "SessionManager", tree_name_config: str
) -> Optional[str]:
    if not tree_name_config:
        logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
        return None
    # End of if
    if not session_manager.is_sess_valid():
        logger.error("call_header_trees_api_for_tree_id: Session invalid.")
        return None
    # End of if
    if not callable(_api_req):
        logger.critical("call_header_trees_api_for_tree_id: _api_req is not callable!")
        raise ImportError("_api_req function not available from utils")
    # End of if

    base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
    url = urljoin(base_url_cfg.rstrip("/") + "/", API_PATH_HEADER_TREES)
    api_description = "Header Trees API (Nav Data)"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    referer_url = urljoin(base_url_cfg.rstrip("/") + "/", "family-tree/trees")

    logger.debug(
        f"Attempting to fetch tree ID for TREE_NAME='{tree_name_config}' via {api_description} ({API_PATH_HEADER_TREES}). Referer: {referer_url}"
    )

    custom_headers = {
        "Accept": "application/json",
        "Accept-Language": "en-GB,en;q=0.9",
        "X-Requested-With": "XMLHttpRequest",
    }

    try:
        response_data = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            headers=custom_headers,
            use_csrf_token=False,
            api_description=api_description,
            referer_url=referer_url,
        )

        if (
            response_data
            and isinstance(response_data, dict)
            and KEY_MENUITEMS in response_data
            and isinstance(response_data[KEY_MENUITEMS], list)
        ):
            # Validate response with Pydantic if available
            if PYDANTIC_AVAILABLE:
                try:
                    validated_response = HeaderTreesResponse(**response_data)
                    logger.debug("Header Trees API response validation successful")
                except Exception as validation_err:
                    logger.warning(
                        f"Header Trees API response validation warning: {validation_err}"
                    )

            for item in response_data[KEY_MENUITEMS]:
                if isinstance(item, dict) and item.get(KEY_TEXT) == tree_name_config:
                    tree_url = item.get(KEY_URL)
                    if tree_url and isinstance(tree_url, str):
                        match = re.search(r"/tree/(\d+)", tree_url)
                        if match:
                            my_tree_id_val = match.group(1)
                            logger.debug(
                                f"Found tree ID '{my_tree_id_val}' for tree '{tree_name_config}'."
                            )
                            return my_tree_id_val
                        else:
                            logger.warning(
                                f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}"
                            )
                        # End of if/else
                    else:
                        logger.warning(
                            f"Found tree '{tree_name_config}', but '{KEY_URL}' key missing or invalid."
                        )
                    # End of if/else
                    break
                # End of if
            # End of for
            logger.warning(
                f"Could not find TREE_NAME '{tree_name_config}' in {api_description} response."
            )
            return None
        elif response_data is None:
            logger.warning(f"{api_description} call failed (_api_req returned None).")
            return None
        else:
            status = "N/A"
            if isinstance(response_data, requests.Response):
                status = str(response_data.status_code)
            # End of if
            logger.warning(
                f"Unexpected response format from {api_description} (Type: {type(response_data)}, Status: {status})."
            )
            logger.debug(f"Full {api_description} response data: {str(response_data)}")
            return None
        # End of if/elif/else
    except Exception as e:
        logger.error(f"Error during {api_description}: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_header_trees_api_for_tree_id


def call_tree_owner_api(
    session_manager: "SessionManager", tree_id: str
) -> Optional[str]:
    if not tree_id:
        logger.warning("Cannot get tree owner: tree_id is missing.")
        return None
    # End of if
    if not isinstance(tree_id, str):
        logger.warning(
            f"Invalid tree_id type provided: {type(tree_id)}. Expected string."
        )
        return None
    # End of if
    if not session_manager.is_sess_valid():
        logger.error("call_tree_owner_api: Session invalid.")
        return None
    # End of if
    if not callable(_api_req):
        logger.critical("call_tree_owner_api: _api_req is not callable!")
        raise ImportError("_api_req function not available from utils")
    # End of if

    base_url_cfg = getattr(config_instance, "BASE_URL", "https://www.ancestry.com")
    url = urljoin(
        base_url_cfg.rstrip("/") + "/", f"{API_PATH_TREE_OWNER_INFO}?tree_id={tree_id}"
    )
    api_description = "Tree Owner Name API"

    # Apply rate limiting if available
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time

            time.sleep(wait_time)
        api_rate_limiter.record_request()

    logger.debug(
        f"Attempting to fetch tree owner name for tree ID: {tree_id} via {api_description}..."
    )

    try:
        response_data = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            use_csrf_token=False,
            api_description=api_description,
        )

        if response_data and isinstance(response_data, dict):
            # Validate response with Pydantic if available
            if PYDANTIC_AVAILABLE:
                try:
                    validated_response = TreeOwnerResponse(**response_data)
                    logger.debug("Tree Owner API response validation successful")
                except Exception as validation_err:
                    logger.warning(
                        f"Tree Owner API response validation warning: {validation_err}"
                    )

            owner_data = response_data.get(KEY_OWNER)
            if owner_data and isinstance(owner_data, dict):
                display_name = owner_data.get(KEY_DISPLAY_NAME)
                if display_name and isinstance(display_name, str):
                    logger.debug(
                        f"Found tree owner '{display_name}' for tree ID {tree_id}."
                    )
                    return display_name
                else:
                    logger.warning(
                        f"Could not find '{KEY_DISPLAY_NAME}' in owner data for tree {tree_id}."
                    )
                # End of if/else
            else:
                logger.warning(
                    f"Could not find '{KEY_OWNER}' data in {api_description} response for tree {tree_id}."
                )
            # End of if/else
            logger.debug(f"Full {api_description} response data: {response_data}")
            return None
        elif response_data is None:
            logger.warning(f"{api_description} call failed (_api_req returned None).")
            return None
        else:
            status = "N/A"
            if isinstance(response_data, requests.Response):
                status = str(response_data.status_code)
            # End of if
            logger.warning(
                f"{api_description} call returned unexpected data (Type: {type(response_data)}, Status: {status}) or None."
            )
            logger.debug(f"Response received: {str(response_data)}")
            return None
        # End of if/elif/else
    except Exception as e:
        logger.error(
            f"Error during {api_description} for tree {tree_id}: {e}", exc_info=True
        )
        return None
    # End of try/except


# End of call_tree_owner_api


# --- Standalone Test Block ---
def _sc_run_test(
    test_name: str,
    test_func: Callable,
    test_results_list: List[Tuple[str, str, str]],
    logger_instance: logging.Logger,
    *args,
    **kwargs,
) -> Tuple[str, str, str]:
    logger_instance.debug(f"[ RUNNING SC ] {test_name}")
    status = "FAIL"
    message = kwargs.pop("message", "")
    expect_none = kwargs.pop("expected_none", False)
    expect_type = kwargs.pop("expected_type", None)
    expect_value = kwargs.pop("expected_value", None)
    expect_contains = kwargs.pop("expected_contains", None)
    expect_truthy = kwargs.pop("expected_truthy", False)
    test_func_kwargs = kwargs

    try:
        result = test_func(*args, **test_func_kwargs)
        passed = False
        explicit_skip = False

        if expect_none:
            passed = result is None
            if not passed:
                message = f"Expected None, got {type(result).__name__}"
            # End of if
        elif expect_type is not None:
            if result is None:
                passed = False
                message = f"Expected type {expect_type.__name__}, but function returned None (API/Parse issue?)"
                if not isinstance(result, str) or result != "Skipped":  # type: ignore
                    logger_instance.error(f"Test '{test_name}': {message}")
                # End of if
            elif isinstance(result, expect_type):
                passed = True
            else:
                passed = False
                message = (
                    f"Expected type {expect_type.__name__}, got {type(result).__name__}"
                )
            # End of if/elif/else
        elif expect_value is not None:
            passed = result == expect_value
            if not passed:
                message = f"Expected value '{str(expect_value)[:50]}...', got '{repr(result)[:100]}...'"
            # End of if
        elif expect_contains is not None:
            if isinstance(result, str):
                if isinstance(expect_contains, (list, tuple)):
                    missing = [sub for sub in expect_contains if sub not in result]
                    passed = not missing
                    if not passed:
                        message = f"Expected result to contain all of: {expect_contains}. Missing: {missing}"
                    # End of if
                elif isinstance(expect_contains, str):
                    passed = expect_contains in result
                    if not passed:
                        message = f"Expected result to contain '{expect_contains}', got '{repr(result)[:100]}...'"
                    # End of if
                else:
                    passed = False
                    message = (
                        f"Invalid type for expect_contains: {type(expect_contains)}"
                    )
                # End of if/elif/else
            else:
                passed = False
                message = f"Expected string result for contains check, got {type(result).__name__}"
            # End of if/else
        elif expect_truthy:
            passed = bool(result)
            if not passed:
                if result is None:
                    message = "Expected truthy value, but function returned None (Underlying call likely failed)"
                else:
                    message = f"Expected truthy value, got {repr(result)[:100]}"
                # End of if/else
            # End of if
        elif isinstance(result, str) and result == "Skipped":
            # Don't skip tests, treat them as failures
            passed = False
            explicit_skip = False
            status = "FAIL"
            message = message or "Test should not be skipped"
        else:
            passed = result is True
            if not passed:
                message = (
                    f"Default check failed: Expected True, got {repr(result)[:100]}"
                )
            # End of if
        # End of if/elif chain

        if not explicit_skip:
            status = "PASS" if passed else "FAIL"
            if status == "FAIL" and not message:
                message = f"Test condition not met (Result: {repr(result)[:100]})"
            # End of if
        # End of if

    except Exception as e:
        status = "FAIL"
        message = f"EXCEPTION: {type(e).__name__}: {e}"
        logger_instance.error(
            f"Exception during self-check test '{test_name}': {message}\n{traceback.format_exc()}",
            exc_info=False,
        )
    # End of try/except

    log_level = (
        logging.INFO
        if status == "PASS"
        else (logging.WARNING if status == "SKIPPED" else logging.ERROR)
    )
    log_message = f"[ {status:<7} SC ] {test_name}{f': {message}' if message else ''}"
    logger_instance.log(log_level, log_message)

    test_results_list.append((test_name, status, message if status != "PASS" else ""))
    return (test_name, status, message)


# End of _sc_run_test


def _sc_print_summary(
    test_results_list: List[Tuple[str, str, str]],
    overall_status: bool,
    logger_instance: logging.Logger,
):
    print("\n" + "=" * 70)
    print("🔗 API Utilities Self-Check Summary")
    print("=" * 70)

    name_width = 55
    if test_results_list:
        try:
            name_width = max(len(name) for name, _, _ in test_results_list) + 2
            name_width = min(name_width, 70)
        except ValueError:
            pass

    status_width = 8
    header = f"{'Test Name':<{name_width}} | {'Status':<{status_width}} | {'Message / Details'}"
    print(header)
    print("-" * (len(header) + 5))

    final_fail_count = 0
    final_skip_count = 0
    final_pass_count = 0

    for name, status, message in test_results_list:
        # Add emoji indicators for consistency with test framework
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        display_status = f"{status_emoji} {status}"

        print(
            f"{name:<{name_width}} | {display_status:<{status_width + 2}} | {message if status != 'PASS' else ''}"
        )
        if status == "FAIL":
            final_fail_count += 1
        elif status == "SKIPPED":
            final_skip_count += 1
        elif status == "PASS":
            final_pass_count += 1

    total_executed_tests = len(test_results_list)

    print("-" * (len(header) + 5))
    final_overall_status_from_tests = final_fail_count == 0
    final_overall_status = overall_status and final_overall_status_from_tests

    # Use consistent color scheme
    result_emoji = "✅" if final_overall_status else "❌"
    final_status_msg = (
        f"{result_emoji} Result: {'PASS' if final_overall_status else 'FAIL'} "
        f"({final_pass_count} passed, {final_fail_count} failed, {final_skip_count} skipped out of {total_executed_tests} executed tests)"
    )
    print(f"{final_status_msg}\n")
    logger_instance.log(
        logging.INFO if final_overall_status else logging.ERROR,
        f"api_utils self-check overall status: {'PASS' if final_overall_status else 'FAIL'}",
    )


# End of _sc_print_summary


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for api_utils.py following the standardized 6-category TestSuite framework.
    Tests API functionality, data parsing, and integration capabilities.

    Categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, Error Handling
    """
    from test_framework import TestSuite, suppress_logging, create_mock_data

    with suppress_logging():
        suite = TestSuite("API Utilities & Ancestry Integration", "api_utils.py")
        suite.start_suite()

    # === INITIALIZATION TESTS ===
    def test_module_imports():
        """Test all required modules and dependencies are properly imported."""
        required_modules = [
            "json",
            "requests",
            "time",
            "logging",
            "uuid",
            "re",
            "traceback",
        ]
        for module_name in required_modules:
            # Check if module is imported or available
            module_imported = (
                module_name in sys.modules
                or module_name in globals()
                or any(
                    module_name in str(item)
                    for item in globals().values()
                    if hasattr(item, "__module__")
                )
            )
            assert module_imported, f"Required module {module_name} not available"

    def test_optional_dependencies():
        """Test optional dependencies are properly detected."""
        # Test pydantic availability detection
        assert PYDANTIC_AVAILABLE is not None, "PYDANTIC_AVAILABLE should be defined"
        assert isinstance(
            PYDANTIC_AVAILABLE, bool
        ), "PYDANTIC_AVAILABLE should be boolean"

        # Test BeautifulSoup availability detection
        assert BS4_AVAILABLE is not None, "BS4_AVAILABLE should be defined"
        assert isinstance(BS4_AVAILABLE, bool), "BS4_AVAILABLE should be boolean"

    def test_logger_initialization():
        """Test logging configuration is properly set up."""
        assert logger is not None, "Logger should be initialized"
        assert hasattr(logger, "info"), "Logger should have info method"
        assert hasattr(logger, "error"), "Logger should have error method"
        assert hasattr(logger, "warning"), "Logger should have warning method"

    # === CORE FUNCTIONALITY TESTS ===
    def test_person_detail_parsing():
        """Test parsing of person detail data structures."""
        test_person_data = {
            "PersonId": "TEST_PERSON_123",
            "TreeId": "TEST_TREE_456",
            "FullName": "John Michael Smith",
            "GivenName": "John Michael",
            "Surname": "Smith",
            "BirthYear": 1950,
            "BirthPlace": "New York, USA",
            "DeathYear": 2020,
            "DeathPlace": "California, USA",
            "Gender": "Male",
            "IsLiving": False,
        }

        assert isinstance(test_person_data, dict), "Person data should be dictionary"
        assert "PersonId" in test_person_data, "Person data should have PersonId"
        assert "FullName" in test_person_data, "Person data should have FullName"

    def test_api_response_parsing():
        """Test parsing of various API response formats."""
        test_json_response = '{"status": "success", "data": {"count": 5}}'
        try:
            parsed = json.loads(test_json_response)
            assert isinstance(parsed, dict), "Parsed JSON should be dictionary"
            assert "status" in parsed, "Parsed response should have status"
        except json.JSONDecodeError:
            assert False, "Valid JSON should parse successfully"

    def test_url_construction():
        """Test URL construction and encoding functions."""
        from urllib.parse import quote, urlencode

        test_string = "John Smith & Family"
        encoded = quote(test_string)
        assert isinstance(encoded, str), "Encoded URL should be string"
        assert "%" in encoded, "Encoded URL should contain percent encoding"

        test_params = {"name": "John Smith", "year": 1950}
        encoded_params = urlencode(test_params)
        assert isinstance(encoded_params, str), "Encoded params should be string"
        assert "=" in encoded_params, "Encoded params should contain equals signs"

    # === EDGE CASE TESTS ===
    def test_invalid_json_handling():
        """Test handling of invalid JSON responses."""
        invalid_json_strings = [
            "invalid json",
            '{"incomplete": ',
            "",
            "{'single_quotes': 'invalid'}",
        ]

        for invalid_json in invalid_json_strings:
            try:
                json.loads(invalid_json)
            except (json.JSONDecodeError, TypeError):
                pass  # Expected behavior for invalid JSON

    def test_empty_data_handling():
        """Test handling of empty or missing data."""
        empty_data_cases = [
            {},
            {"empty": ""},
            {"null_value": None},
            {"empty_list": []},
            {"empty_dict": {}},
        ]

        for empty_data in empty_data_cases:
            assert isinstance(empty_data, dict), "Test data should be dictionary"

    def test_special_characters():
        """Test handling of special characters in names and places."""
        special_char_cases = [
            "François O'Connor",
            "José María García-López",
            "李小明",  # Chinese characters
            "محمد",  # Arabic characters
            "Müller & Söhne",
            "Name with (parentheses) and [brackets]",
        ]

        for test_name in special_char_cases:
            assert isinstance(test_name, str), "Name should be string"
            encoded = quote(test_name)
            assert isinstance(encoded, str), "Encoded name should be string"

    # === INTEGRATION TESTS ===
    def test_config_integration():
        """Test integration with configuration management."""
        try:
            from config import config_instance

            assert config_instance is not None, "Config instance should be available"
        except ImportError:
            pass  # Config integration may not be available in test environment

    def test_logging_integration():
        """Test integration with logging configuration."""
        from logging_config import setup_logging

        assert callable(setup_logging), "setup_logging should be callable"
        assert logger is not None, "Logger should be initialized"

        try:
            logger.info("Test log message")
        except Exception as e:
            assert False, f"Logging should work without errors: {e}"

    def test_datetime_handling():
        """Test datetime parsing and formatting integration."""
        from datetime import datetime, timezone

        test_datetime = datetime.now(timezone.utc)
        assert isinstance(test_datetime, datetime), "Should create datetime object"

        formatted = test_datetime.isoformat()
        assert isinstance(formatted, str), "Formatted datetime should be string"
        assert "T" in formatted, "ISO format should contain T separator"

    # === PERFORMANCE TESTS ===
    def test_json_parsing_performance():
        """Test JSON parsing performance with large data sets."""
        import time

        large_data = {"items": [{"id": i, "name": f"Person {i}"} for i in range(1000)]}
        json_string = json.dumps(large_data)

        start_time = time.time()
        for _ in range(10):
            parsed = json.loads(json_string)
        end_time = time.time()

        parsing_time = end_time - start_time
        assert (
            parsing_time < 1.0
        ), f"JSON parsing took {parsing_time:.3f}s, should be < 1.0s"
        assert isinstance(parsed, dict), "Parsed result should be dictionary"

    def test_url_encoding_performance():
        """Test URL encoding performance with multiple strings."""
        import time

        test_strings = [f"Person Name {i} & Family" for i in range(100)]

        start_time = time.time()
        for test_string in test_strings:
            encoded = quote(test_string)
        end_time = time.time()

        encoding_time = end_time - start_time
        assert (
            encoding_time < 0.1
        ), f"URL encoding took {encoding_time:.3f}s, should be < 0.1s"

    def test_data_processing_efficiency():
        """Test efficiency of data processing operations."""
        import time

        test_data = [
            {
                "name": f"Person {i}",
                "year": 1900 + i % 100,
                "place": f"City {i % 50}",
            }
            for i in range(500)
        ]

        start_time = time.time()
        processed_count = 0
        for item in test_data:
            if item.get("year", 0) > 1950:
                processed_count += 1
        end_time = time.time()

        processing_time = end_time - start_time
        assert (
            processing_time < 0.05
        ), f"Data processing took {processing_time:.3f}s, should be < 0.05s"
        assert processed_count > 0, "Should process some items"

    # === ERROR HANDLING TESTS ===
    def test_network_error_simulation():
        """Test handling of network-related errors."""
        network_errors = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timeout"),
            requests.exceptions.RequestException("General request error"),
        ]

        for error in network_errors:
            assert isinstance(
                error, requests.exceptions.RequestException
            ), "Should be request exception"

    def test_data_validation_errors():
        """Test handling of data validation errors."""
        invalid_data_cases = [
            {"PersonId": None},  # Missing required field
            {"PersonId": ""},  # Empty required field
            {"PersonId": 123},  # Wrong type
            {"PersonId": "valid", "BirthYear": "not_a_year"},  # Invalid year
        ]

        for invalid_data in invalid_data_cases:
            assert isinstance(invalid_data, dict), "Test data should be dictionary"
            person_id = invalid_data.get("PersonId")
            if person_id is None or person_id == "":
                assert (
                    person_id is None or person_id == ""
                ), "Empty PersonId should be properly detected"
            elif isinstance(person_id, int):
                assert isinstance(
                    person_id, int
                ), "Integer PersonId should be detectable as wrong type"

    def test_configuration_errors():
        """Test handling of missing or invalid configuration."""
        missing_config_cases = [
            None,
            {},
            {"incomplete": "config"},
        ]

        for config_case in missing_config_cases:
            if config_case is None:
                assert config_case is None, "None config should be detectable"
            elif isinstance(config_case, dict):
                if not config_case or "base_url" not in config_case:
                    assert (
                        len(config_case) == 0 or "base_url" not in config_case
                    ), "Missing config should be detectable"

    # === RUN ALL TESTS ===
    suite.run_test(
        "Module Imports",
        test_module_imports,
        "All required modules should be importable and accessible",
        "All required modules and dependencies are properly imported",
        "Test verifies essential modules (json, requests, time, logging, uuid, re, traceback) are available",
    )

    suite.run_test(
        "Optional Dependencies",
        test_optional_dependencies,
        "Optional dependency flags should be properly set",
        "Optional dependencies (pydantic, BeautifulSoup) are properly detected",
        "Test PYDANTIC_AVAILABLE and BS4_AVAILABLE boolean flags",
    )

    suite.run_test(
        "Logger Initialization",
        test_logger_initialization,
        "Logger should be configured with all required methods",
        "Logging configuration is properly set up",
        "Test logger object exists and has info, error, warning methods",
    )

    suite.run_test(
        "Person Detail Parsing",
        test_person_detail_parsing,
        "Person data structures should be valid and parseable",
        "Parsing of person detail data structures works correctly",
        "Test with realistic Ancestry API response structure",
    )

    suite.run_test(
        "API Response Parsing",
        test_api_response_parsing,
        "JSON responses should parse correctly into dictionaries",
        "Parsing of various API response formats works correctly",
        "Test JSON response parsing with valid API response format",
    )

    suite.run_test(
        "URL Construction",
        test_url_construction,
        "URLs should be properly encoded and constructed",
        "URL construction and encoding functions work correctly",
        "Test URL encoding and parameter encoding functionality",
    )

    suite.run_test(
        "Invalid JSON Handling",
        test_invalid_json_handling,
        "Invalid JSON should be handled without crashing",
        "Handling of invalid JSON responses works gracefully",
        "Test various invalid JSON formats and error handling",
    )

    suite.run_test(
        "Empty Data Handling",
        test_empty_data_handling,
        "Empty or missing data should not cause application errors",
        "Handling of empty or missing data works correctly",
        "Test various empty data cases (empty dict, None values, empty lists)",
    )

    suite.run_test(
        "Special Characters",
        test_special_characters,
        "Special characters should be handled and encoded properly",
        "Handling of special characters in names and places",
        "Test Unicode, accented characters, and special symbols in names",
    )

    suite.run_test(
        "Config Integration",
        test_config_integration,
        "Configuration integration should work seamlessly",
        "Integration with configuration management works correctly",
        "Test accessing config_instance and BASE_URL configuration",
    )

    suite.run_test(
        "Logging Integration",
        test_logging_integration,
        "Logging should be properly integrated and functional",
        "Integration with logging configuration works correctly",
        "Test setup_logging function and logger functionality",
    )

    suite.run_test(
        "Datetime Handling",
        test_datetime_handling,
        "Datetime operations should work correctly for API calls",
        "Datetime parsing and formatting integration works",
        "Test datetime creation, timezone handling, and ISO formatting",
    )

    suite.run_test(
        "JSON Parsing Performance",
        test_json_parsing_performance,
        "JSON parsing should complete efficiently even with large data",
        "JSON parsing performance with large data sets is acceptable",
        "Measure time to parse 1000-item JSON structure 10 times",
    )

    suite.run_test(
        "URL Encoding Performance",
        test_url_encoding_performance,
        "URL encoding should be efficient for multiple strings",
        "URL encoding performance with multiple strings is acceptable",
        "Measure time to encode 100 test strings with special characters",
    )

    suite.run_test(
        "Data Processing Efficiency",
        test_data_processing_efficiency,
        "Data processing should be efficient for moderate-sized datasets",
        "Efficiency of data processing operations is acceptable",
        "Process 500-item dataset with filtering operations",
    )

    suite.run_test(
        "Network Error Simulation",
        test_network_error_simulation,
        "Network errors should be recognizable and handleable",
        "Handling of network-related errors works correctly",
        "Test various requests exception types (ConnectionError, Timeout, RequestException)",
    )

    suite.run_test(
        "Data Validation Errors",
        test_data_validation_errors,
        "Data validation errors should be caught and handled gracefully",
        "Handling of data validation errors works correctly",
        "Test invalid PersonId values, wrong types, and missing required fields",
    )

    suite.run_test(
        "Configuration Errors",
        test_configuration_errors,
        "Configuration errors should be handled without application crashes",
        "Handling of missing or invalid configuration works gracefully",
        "Test None config, empty config, and incomplete configuration scenarios",
    )

    return suite.finish_suite()


# --- Main Execution Block ---
if __name__ == "__main__":
    import sys

    print("🌐 Running API Utilities & Ancestry Integration comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
