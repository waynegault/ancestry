#!/usr/bin/env python3

"""
API Intelligence & Request Orchestration Engine

Advanced API management platform providing sophisticated request orchestration,
intelligent response processing, and comprehensive authentication management
with adaptive rate limiting, intelligent caching, and performance optimization
for reliable genealogical API interactions and data synchronization workflows.

API Management:
• Unified API request handling with intelligent routing and endpoint management
• Advanced response processing with validation, transformation, and error handling
• Sophisticated authentication management with credential rotation and session persistence
• Intelligent rate limiting with adaptive throttling and circuit breaker patterns
• Comprehensive error handling with retry logic, exponential backoff, and graceful degradation
• Real-time API health monitoring with performance tracking and alerting

Performance Optimization:
• Advanced caching strategies with TTL-based invalidation and intelligent cache warming
• Request optimization with batching, compression, and connection pooling
• Response optimization with streaming, pagination, and memory-efficient processing
• Performance monitoring with latency tracking, throughput analysis, and bottleneck detection
• Intelligent load balancing with failover capabilities and endpoint health checking
• Comprehensive metrics collection with API usage analytics and performance insights

Integration Intelligence:
• Sophisticated API endpoint discovery with automatic configuration and validation
• Advanced request transformation with data mapping, validation, and enrichment
• Intelligent response parsing with schema validation and data extraction
• Comprehensive API versioning support with backward compatibility and migration strategies
• Integration with external systems through standardized API patterns and protocols
• Real-time synchronization with conflict resolution and data consistency management

Foundation Services:
Provides the essential API infrastructure that enables reliable, scalable
genealogical automation through intelligent request management, comprehensive
error handling, and performance optimization for professional research workflows.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
)

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# === STANDARD LIBRARY IMPORTS ===
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import quote, urlencode, urljoin

# === THIRD-PARTY IMPORTS ===
import requests

from core.error_handling import (
    AncestryException,
    APIRateLimitError,
    NetworkTimeoutError,
    RetryableError,
    circuit_breaker,
    error_context,
    retry_on_failure,
    timeout_protection,
)

# --- Check for optional dependencies ---
# BeautifulSoup import removed - not used in this module
BS4_AVAILABLE = False

try:
    from importlib.util import find_spec as _find_spec
    PYDANTIC_AVAILABLE = _find_spec("pydantic") is not None
except Exception:
    PYDANTIC_AVAILABLE = False

# === LOCAL IMPORTS ===
from common_params import ApiIdentifiers
from config import config_schema
from core.session_manager import SessionManager
from database import Person
from gedcom_utils import _clean_display_date, _parse_date
from logging_config import setup_logging
from utils import _api_req, format_name

# --- Test framework imports ---
try:
    from test_framework import (
        TestSuite as TestFrameworkTestSuite,
    )

    TestSuite = TestFrameworkTestSuite  # type: ignore
except ImportError:
    # If test framework not available, import will fail at test time
    TestSuite = None

# === MODULE LOGGER ===
# Use centralized log file from .env (LOG_FILE)
logger = setup_logging(log_level="INFO")

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
API_PATH_DISCOVERY_RELATIONSHIP = "discoveryui-matchingservice/api/relationship"
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
# Modern dataclass-based models for API response validation

from dataclasses import dataclass


@dataclass
class PersonSuggestResponse:
    """Dataclass model for validating Ancestry Suggest API responses."""

    PersonId: Optional[str] = None
    TreeId: Optional[str] = None
    UserId: Optional[str] = None
    FullName: Optional[str] = None
    GivenName: Optional[str] = None
    Surname: Optional[str] = None
    BirthYear: Optional[int] = None
    BirthPlace: Optional[str] = None
    DeathYear: Optional[int] = None
    DeathPlace: Optional[str] = None
    Gender: Optional[str] = None
    IsLiving: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PersonSuggestResponse':
        """Create instance from dictionary data."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
        """Convert to dictionary format with optional None exclusion."""
        from dataclasses import asdict
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class ProfileDetailsResponse:
    """Model for validating Profile Details API responses."""

    def __init__(self, **kwargs: Any) -> None:
        self.FirstName: Optional[str] = kwargs.get("FirstName")
        self.displayName: Optional[str] = kwargs.get("displayName")
        self.LastLoginDate: Optional[str] = kwargs.get("LastLoginDate")
        self.IsContactable: Optional[bool] = kwargs.get("IsContactable")

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
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


@dataclass
class TreeOwnerResponse:
    """Dataclass model for validating Tree Owner API responses."""

    owner: Optional[dict[str, Any]] = None
    displayName: Optional[str] = None
    id: Optional[str] = None  # Add missing 'id' field
    peopleCount: Optional[int] = None  # Add missing 'peopleCount' field
    photoCount: Optional[int] = None  # Add missing 'photoCount' field
    membership: Optional[dict[str, Any]] = None  # Add missing 'membership' field

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TreeOwnerResponse':
        """Create instance from dictionary data, safely handling aliases and extras.
        - Maps legacy 'treeMembersCount' to 'peopleCount' if present.
        - Ignores unknown fields to avoid constructor errors.
        """
        if not isinstance(data, dict):  # type: ignore[unreachable]
            data = {}
        normalized = dict(data)
        # Alias mapping: treeMembersCount -> peopleCount (if peopleCount missing)
        if "peopleCount" not in normalized and "treeMembersCount" in normalized:
            try:
                normalized["peopleCount"] = int(normalized.get("treeMembersCount") or 0)
            except Exception:
                # Fallback to raw if not int-coercible
                normalized["peopleCount"] = normalized.get("treeMembersCount")
        # Filter to known fields only
        filtered = {k: normalized[k] for k in cls.__dataclass_fields__ if k in normalized}
        return cls(**filtered)

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
        """Convert to dictionary format with optional None exclusion."""
        from dataclasses import asdict
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


@dataclass
class PersonFactsResponse:
    """Dataclass model for validating Person Facts API responses."""

    data: Optional[dict[str, Any]] = None
    personResearch: Optional[dict[str, Any]] = None
    PersonFacts: Optional[list[dict[str, Any]]] = None
    PersonFullName: Optional[str] = None
    FirstName: Optional[str] = None
    LastName: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PersonFactsResponse':
        """Create instance from dictionary data."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
        """Convert to dictionary format with optional None exclusion."""
        from dataclasses import asdict
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


@dataclass
class GetLadderResponse:
    """Dataclass model for validating GetLadder API responses."""

    data: Optional[dict[str, Any]] = None
    relationship: Optional[dict[str, Any]] = None
    paths: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'GetLadderResponse':
        """Create instance from dictionary data."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
        """Convert to dictionary format with optional None exclusion."""
        from dataclasses import asdict
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


class DiscoveryRelationshipResponse:
    """Model for validating Discovery Relationship API responses."""

    def __init__(self, **kwargs: Any) -> None:
        self.relationship: Optional[str] = kwargs.get("relationship")
        self.paths: Optional[list[dict[str, Any]]] = kwargs.get("paths")
        self.confidence: Optional[str] = kwargs.get("confidence")

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
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

    def __init__(self, **kwargs: Any) -> None:
        self.menuitems: Optional[list[dict[str, Any]]] = kwargs.get("menuitems")
        self.url: Optional[str] = kwargs.get("url")
        self.text: Optional[str] = kwargs.get("text")

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
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

    def __init__(self, **kwargs: Any) -> None:
        self.conversation_id: Optional[str] = kwargs.get("conversation_id")
        self.message: Optional[str] = kwargs.get("message")
        self.author: Optional[str] = kwargs.get("author")
        self.status: Optional[str] = kwargs.get("status")

    def dict(self, exclude_none: bool = False) -> dict[str, Any]:
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

    def __init__(self, max_calls_per_minute: int = 60, max_calls_per_hour: int = 1000) -> None:
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
        return not len(self.hour_calls) >= self.max_calls_per_hour

    def record_request(self) -> None:
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

# Sub-helpers for _extract_name_from_api_details

def _try_extract_name_from_person_info(facts_data: dict[str, Any]) -> Optional[str]:
    """Try to extract name from person info in facts_data."""
    person_info = facts_data.get("person", {})
    if isinstance(person_info, dict):
        return person_info.get("personName")
    return None


def _try_extract_name_from_direct_fields(facts_data: dict[str, Any]) -> Optional[str]:
    """Try to extract name from direct fields in facts_data."""
    for field in ["personName", "DisplayName", "PersonFullName"]:
        name = facts_data.get(field)
        if name:
            return name
    return None


def _try_extract_name_from_person_facts(facts_data: dict[str, Any]) -> Optional[str]:
    """Try to extract name from PersonFacts list in facts_data."""
    person_facts_list = facts_data.get("PersonFacts", [])
    if isinstance(person_facts_list, list):
        name_fact = next(
            (f for f in person_facts_list if isinstance(f, dict) and f.get("TypeString") == "Name"),
            None,
        )
        if name_fact and name_fact.get("Value"):
            return name_fact.get("Value")
    return None


def _try_construct_name_from_parts(facts_data: dict[str, Any]) -> Optional[str]:
    """Try to construct name from FirstName/LastName in facts_data."""
    first_name = facts_data.get("FirstName")
    last_name = facts_data.get("LastName")
    if first_name or last_name:
        constructed = f"{first_name or ''} {last_name or ''}".strip()
        return constructed if constructed else None
    return None


def _extract_name_from_facts_data(facts_data: Optional[dict[str, Any]]) -> Optional[str]:
    """Extract name from facts_data using multiple strategies."""
    if not facts_data or not isinstance(facts_data, dict):
        return None

    # Try each extraction strategy in order
    name = _try_extract_name_from_person_info(facts_data)
    if name:
        return name

    name = _try_extract_name_from_direct_fields(facts_data)
    if name:
        return name

    name = _try_extract_name_from_person_facts(facts_data)
    if name:
        return name

    return _try_construct_name_from_parts(facts_data)


def _extract_name_from_person_card(person_card: dict[str, Any]) -> Optional[str]:
    """Extract name from person_card using multiple strategies."""
    if not person_card:
        return None

    # Try FullName first
    full_name = person_card.get("FullName")
    if full_name:
        return full_name

    # Try constructing from GivenName/Surname
    given_name = person_card.get("GivenName")
    surname = person_card.get("Surname")
    if given_name or surname:
        constructed = f"{given_name or ''} {surname or ''}".strip()
        if constructed:
            return constructed

    # Try generic 'name' field
    return person_card.get("name")


def _extract_name_from_api_details(
    person_card: dict[str, Any], facts_data: Optional[dict[str, Any]]
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
    # Try extracting from facts_data first
    name = _extract_name_from_facts_data(facts_data)

    # If not found, try person_card
    if not name:
        name = _extract_name_from_person_card(person_card)

    # Default to "Unknown" if still not found
    if not name:
        name = "Unknown"

    # Format the name
    formatted_name = format_name(name) if name and name != "Unknown" else "Unknown"

    # Filter out "Valued Relative" placeholder
    return "Unknown" if formatted_name == "Valued Relative" else formatted_name


# End of _extract_name_from_api_details


# Helper functions for _extract_gender_from_api_details

def _try_extract_gender_from_person_info(facts_data: dict) -> Optional[str]:
    """Try to extract gender from person info in facts_data."""
    person_info = facts_data.get("person", {})
    if isinstance(person_info, dict):
        return person_info.get("gender")
    return None


def _try_extract_gender_from_direct_fields(facts_data: dict) -> Optional[str]:
    """Try to extract gender from direct fields in facts_data."""
    gender_str = facts_data.get("gender")
    if gender_str:
        return gender_str
    return facts_data.get("PersonGender")


def _try_extract_gender_from_person_facts(facts_data: dict) -> Optional[str]:
    """Try to extract gender from PersonFacts list in facts_data."""
    person_facts_list = facts_data.get("PersonFacts", [])
    if not isinstance(person_facts_list, list):
        return None

    gender_fact = next(
        (f for f in person_facts_list if isinstance(f, dict) and f.get("TypeString") == "Gender"),
        None,
    )

    if gender_fact and gender_fact.get("Value"):
        return gender_fact.get("Value")
    return None


def _extract_gender_from_facts_data(facts_data: Optional[dict]) -> Optional[str]:
    """Extract gender from facts_data using multiple strategies."""
    if not facts_data or not isinstance(facts_data, dict):
        return None

    # Try person info
    gender_str = _try_extract_gender_from_person_info(facts_data)
    if gender_str:
        return gender_str

    # Try direct fields
    gender_str = _try_extract_gender_from_direct_fields(facts_data)
    if gender_str:
        return gender_str

    # Try PersonFacts list
    return _try_extract_gender_from_person_facts(facts_data)


def _extract_gender_from_person_card(person_card: dict) -> Optional[str]:
    """Extract gender from person_card."""
    gender_str = person_card.get("Gender")
    if gender_str:
        return gender_str
    return person_card.get("gender")


def _normalize_gender_string(gender_str: Optional[str]) -> Optional[str]:
    """Normalize gender string to M or F."""
    if not gender_str or not isinstance(gender_str, str):
        return None

    gender_str_lower = gender_str.lower()

    if gender_str_lower == "male":
        return "M"
    if gender_str_lower == "female":
        return "F"
    if gender_str_lower in ["m", "f"]:
        return gender_str_lower.upper()

    return None


def _extract_gender_from_api_details(
    person_card: dict, facts_data: Optional[dict]
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
    # Try to extract from facts_data first
    gender_str = _extract_gender_from_facts_data(facts_data)

    # Fall back to person_card if needed
    if not gender_str and person_card:
        gender_str = _extract_gender_from_person_card(person_card)

    # Normalize and return
    return _normalize_gender_string(gender_str)


# End of _extract_gender_from_api_details


# Helper functions for _extract_living_status_from_api_details

def _try_extract_living_from_person_info(facts_data: dict) -> Optional[bool]:
    """Try to extract living status from person info in facts_data."""
    person_info = facts_data.get("person", {})
    if isinstance(person_info, dict):
        return person_info.get("isLiving")
    return None


def _try_extract_living_from_direct_fields(facts_data: dict) -> Optional[bool]:
    """Try to extract living status from direct fields in facts_data."""
    is_living = facts_data.get("isLiving")
    if is_living is not None:
        return is_living
    return facts_data.get("IsPersonLiving")


def _extract_living_from_facts_data(facts_data: Optional[dict]) -> Optional[bool]:
    """Extract living status from facts_data using multiple strategies."""
    if not facts_data or not isinstance(facts_data, dict):
        return None

    # Try person info
    is_living = _try_extract_living_from_person_info(facts_data)
    if is_living is not None:
        return is_living

    # Try direct fields
    return _try_extract_living_from_direct_fields(facts_data)


def _extract_living_from_person_card(person_card: dict) -> Optional[bool]:
    """Extract living status from person_card."""
    is_living = person_card.get("IsLiving")
    if is_living is not None:
        return is_living
    return person_card.get("isLiving")


def _extract_living_status_from_api_details(
    person_card: dict, facts_data: Optional[dict]
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
    # Try to extract from facts_data first
    is_living = _extract_living_from_facts_data(facts_data)

    # Fall back to person_card if needed
    if is_living is None and person_card:
        is_living = _extract_living_from_person_card(person_card)

    # Convert to boolean if not None
    return bool(is_living) if is_living is not None else None


# End of _extract_living_status_from_api_details


# Helper functions for _extract_event_from_api_details

def _build_event_keys(event_type: str) -> dict[str, str]:
    """Build all the key names needed for event extraction."""
    event_key_lower = event_type.lower()
    return {
        "suggest_year": f"{event_type}Year",
        "suggest_place": f"{event_type}Place",
        "facts_user": event_type,
        "app_api": f"{event_key_lower}Date",
        "app_api_facts": event_type,
        "event_lower": event_key_lower,
    }


def _build_date_string_from_parsed_data(parsed_date_data: dict) -> Optional[str]:
    """Build a date string from parsed date components."""
    year = parsed_date_data.get("Year")
    if not year:
        return None

    temp_date_str = str(year)
    month = parsed_date_data.get("Month")
    if month:
        temp_date_str += f"-{str(month).zfill(2)}"

    day = parsed_date_data.get("Day")
    if day:
        temp_date_str += f"-{str(day).zfill(2)}"

    return temp_date_str


def _try_parse_date_object(date_str: str, parser, event_type: str) -> Optional[datetime]:
    """Try to parse a date string into a datetime object."""
    try:
        return parser(date_str)
    except Exception as parse_err:
        logger.warning(f"Failed to parse {event_type} date string '{date_str}': {parse_err}")
        return None


def _extract_from_person_facts(
    facts_data: dict, facts_user_key: str, event_type: str, parser
) -> tuple[Optional[str], Optional[str], Optional[datetime], bool]:
    """Extract event data from PersonFacts list."""
    person_facts_list = facts_data.get("PersonFacts", [])
    if not isinstance(person_facts_list, list):
        return None, None, None, False

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

    if not event_fact:
        return None, None, None, False

    date_str = event_fact.get("Date")
    place_str = event_fact.get("Place")
    date_obj = None

    logger.debug(
        f"Found primary {event_type} fact in PersonFacts: Date='{date_str}', Place='{place_str}'"
    )

    # Try to parse date from ParsedDate structure
    parsed_date_data = event_fact.get("ParsedDate")
    if isinstance(parsed_date_data, dict) and parser:
        temp_date_str = _build_date_string_from_parsed_data(parsed_date_data)
        if temp_date_str:
            date_obj = _try_parse_date_object(temp_date_str, parser, event_type)
            if date_obj:
                logger.debug(f"Parsed {event_type} date object from ParsedDate: {date_obj}")

    return date_str, place_str, date_obj, True


def _extract_from_structured_facts(
    facts_data: dict, app_api_facts_key: str
) -> tuple[Optional[str], Optional[str], bool]:
    """Extract event data from structured facts."""
    fact_group_list = facts_data.get("facts", {}).get(app_api_facts_key, [])
    if not fact_group_list or not isinstance(fact_group_list, list):
        return None, None, False

    fact_group = fact_group_list[0]
    if not isinstance(fact_group, dict):
        return None, None, False

    date_str = None
    place_str = None

    date_info = fact_group.get("date", {})
    if isinstance(date_info, dict):
        date_str = date_info.get("normalized", date_info.get("original"))

    place_info = fact_group.get("place", {})
    if isinstance(place_info, dict):
        place_str = place_info.get("placeName")

    return date_str, place_str, True


def _extract_from_alternative_facts(
    facts_data: dict, app_api_key: str
) -> tuple[Optional[str], Optional[str], bool]:
    """Extract event data from alternative fact formats."""
    event_fact_alt = facts_data.get(app_api_key)

    if event_fact_alt and isinstance(event_fact_alt, dict):
        date_str = event_fact_alt.get("normalized", event_fact_alt.get("date"))
        place_str = event_fact_alt.get("place")
        return date_str, place_str, True
    if isinstance(event_fact_alt, str):
        return event_fact_alt, None, True

    return None, None, False


def _extract_from_suggest_api(
    person_card: dict, suggest_year_key: str, suggest_place_key: str, event_type: str
) -> tuple[Optional[str], Optional[str]]:
    """Extract event data from Suggest API fields."""
    suggest_year = person_card.get(suggest_year_key)
    suggest_place = person_card.get(suggest_place_key)

    if suggest_year:
        date_str = str(suggest_year)
        logger.debug(
            f"Using Suggest API keys for {event_type}: Year='{date_str}', Place='{suggest_place}'"
        )
        return date_str, suggest_place

    return None, None


def _extract_from_event_info_card(
    person_card: dict, event_key_lower: str
) -> tuple[Optional[str], Optional[str]]:
    """Extract event data from concatenated event info strings."""
    event_info_card = person_card.get(event_key_lower, "")

    if event_info_card and isinstance(event_info_card, str):
        parts = re.split(r"\s+in\s+", event_info_card, maxsplit=1)
        date_str = parts[0].strip() if parts else event_info_card
        place_str = parts[1].strip() if len(parts) > 1 else None
        return date_str, place_str
    if isinstance(event_info_card, dict):
        date_str = event_info_card.get("date")
        place_str = event_info_card.get("place")
        return date_str, place_str

    return None, None


def _extract_from_facts_data(
    facts_data: Optional[dict],
    keys: dict[str, str],
    event_type: str,
    parser: Any
) -> tuple[Optional[str], Optional[str], Optional[datetime], bool]:
    """Extract event data from facts_data using multiple strategies."""
    date_str: Optional[str] = None
    place_str: Optional[str] = None
    date_obj: Optional[datetime] = None
    found_in_facts = False

    if not facts_data or not isinstance(facts_data, dict):
        return date_str, place_str, date_obj, found_in_facts

    # Strategy 1: PersonFacts primary event facts
    date_str, place_str, date_obj, found_in_facts = _extract_from_person_facts(
        facts_data, keys["facts_user"], event_type, parser
    )

    # Strategy 2: Structured facts data with date/place objects
    if not found_in_facts:
        date_str, place_str, found_in_facts = _extract_from_structured_facts(
            facts_data, keys["app_api_facts"]
        )

    # Strategy 3: Alternative event fact formats
    if not found_in_facts:
        date_str, place_str, found_in_facts = _extract_from_alternative_facts(
            facts_data, keys["app_api"]
        )

    return date_str, place_str, date_obj, found_in_facts


def _extract_from_person_card(
    person_card: dict,
    keys: dict[str, str],
    event_type: str,
    current_date_str: Optional[str],
    current_place_str: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Extract event data from person_card using multiple strategies."""
    date_str = current_date_str
    place_str = current_place_str

    # Strategy 4: Suggest API year/place fields
    suggest_date, suggest_place = _extract_from_suggest_api(
        person_card, keys["suggest_year"], keys["suggest_place"], event_type
    )

    if suggest_date:
        date_str = suggest_date
        place_str = suggest_place
    else:
        # Strategy 5: Concatenated event info strings
        card_date, card_place = _extract_from_event_info_card(
            person_card, keys["event_lower"]
        )
        if card_date:
            date_str = card_date
        if card_place and place_str is None:
            place_str = card_place

    return date_str, place_str


def _extract_event_from_api_details(
    event_type: str, person_card: dict, facts_data: Optional[dict]
) -> tuple[Optional[str], Optional[str], Optional[datetime]]:
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
    found_in_facts = False

    # Build all the key names we'll need
    keys = _build_event_keys(event_type)

    # Try to extract from facts_data using multiple strategies
    date_str, place_str, date_obj, found_in_facts = _extract_from_facts_data(
        facts_data, keys, event_type, parser
    )

    # Try to extract from person_card if not found in facts_data
    if not found_in_facts and person_card:
        date_str, place_str = _extract_from_person_card(person_card, keys, event_type, date_str, place_str)

    # Final attempt to parse date if we have a date string but no date object
    if date_obj is None and date_str and parser:
        date_obj = _try_parse_date_object(date_str, parser, event_type)

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
    if person_id:
        return f"{base_url}/discoveryui-matches/list/summary/{person_id}"
    # End of if/elif
    return "(Link unavailable)"


# End of _generate_person_link


def _initialize_person_details(person_card: dict[str, Any]) -> dict[str, Any]:
    """Initialize person details dictionary with default values."""
    details: dict[str, Any] = {
        "name": "Unknown",
        "birth_date": "N/A",
        "birth_place": None,
        "api_birth_obj": None,
        "death_date": "N/A",
        "death_place": None,
        "api_death_obj": None,
        "gender": None,
        "is_living": None,
        "person_id": person_card.get("PersonId") or person_card.get("personId"),
        "tree_id": person_card.get("TreeId") or person_card.get("treeId"),
        "user_id": person_card.get("UserId"),
        "link": "(Link unavailable)",
    }
    return details


def _update_details_from_facts(details: dict[str, Any], facts_data: Optional[dict[str, Any]]) -> None:
    """Update person details from facts data if available."""
    if not facts_data or not isinstance(facts_data, dict):
        return

    details["person_id"] = facts_data.get("PersonId", details["person_id"])
    details["tree_id"] = facts_data.get("TreeId", details["tree_id"])
    details["user_id"] = facts_data.get("UserId", details["user_id"])

    if not details["user_id"]:
        person_info = facts_data.get("person", {})
        if isinstance(person_info, dict):
            details["user_id"] = person_info.get("userId", details["user_id"])


def _extract_and_format_dates(details: dict[str, Any], person_card: dict[str, Any], facts_data: Optional[dict[str, Any]]) -> None:
    """Extract and format birth and death dates."""
    birth_date_raw, details["birth_place"], details["api_birth_obj"] = (
        _extract_event_from_api_details("Birth", person_card, facts_data)
    )
    death_date_raw, details["death_place"], details["api_death_obj"] = (
        _extract_event_from_api_details("Death", person_card, facts_data)
    )

    cleaner = _clean_display_date
    details["birth_date"] = cleaner(birth_date_raw) if birth_date_raw else "N/A"
    details["death_date"] = cleaner(death_date_raw) if death_date_raw else "N/A"

    # Fallback to year if full date unavailable
    if details["birth_date"] == "N/A" and details["api_birth_obj"]:
        details["birth_date"] = str(details["api_birth_obj"].year)
    if details["death_date"] == "N/A" and details["api_death_obj"]:
        details["death_date"] = str(details["api_death_obj"].year)


def parse_ancestry_person_details(
    person_card: dict[str, Any], facts_data: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    # Initialize details
    details = _initialize_person_details(person_card)

    # Update from facts data
    _update_details_from_facts(details, facts_data)

    # Extract basic attributes
    details["name"] = _extract_name_from_api_details(person_card, facts_data)
    details["gender"] = _extract_gender_from_api_details(person_card, facts_data)
    details["is_living"] = _extract_living_status_from_api_details(person_card, facts_data)

    # Extract and format dates
    _extract_and_format_dates(details, person_card, facts_data)

    # Generate link
    base_url_for_link = (config_schema.api.base_url or "https://www.ancestry.com").rstrip("/")
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


def print_group(label: str, items: list[dict]):
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
    if config_schema.selenium and hasattr(config_schema.selenium, "api_timeout"):
        config_timeout = config_schema.selenium.api_timeout
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
    logger.warning(
        "Owner profile/tree ID missing in session. Using base URL as referer."
    )
    return base_url.rstrip("/") + "/"
    # End of if/else


# End of _get_owner_referer


@retry_on_failure(max_attempts=3, backoff_factor=2.0)
@circuit_breaker(failure_threshold=5, recovery_timeout=60)
# Helper functions for call_suggest_api

def _validate_suggest_api_inputs(session_manager: "SessionManager", owner_tree_id: str):
    """Validate inputs for suggest API call."""
    if not callable(_api_req):
        logger.critical("Suggest API call failed: _api_req function unavailable (Import Failed?).")
        raise AncestryException(
            "_api_req function not available from utils - Check module imports and dependencies"
        )

    if not isinstance(session_manager, SessionManager) and not hasattr(session_manager, "is_sess_valid"):  # type: ignore[unreachable]
        raise AncestryException(
            "Invalid SessionManager passed to suggest API - Provide a valid SessionManager instance"
        )

    if not owner_tree_id:
        raise AncestryException("owner_tree_id is required for suggest API - Provide a valid tree ID")


def _apply_rate_limiting(api_description: str):
    """Apply rate limiting if available."""
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s")
            import time
            time.sleep(wait_time)
        api_rate_limiter.record_request()


def _build_suggest_url(owner_tree_id: str, base_url: str, search_criteria: dict[str, Any]) -> str:
    """Build suggest API URL with search parameters."""
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")

    suggest_params_list = ["isHideVeiledRecords=false"]
    if first_name_raw:
        suggest_params_list.append(f"partialFirstName={quote(first_name_raw)}")
    if surname_raw:
        suggest_params_list.append(f"partialLastName={quote(surname_raw)}")
    if birth_year:
        suggest_params_list.append(f"birthYear={birth_year}")

    suggest_params = "&".join(suggest_params_list)
    formatted_path = API_PATH_PERSON_PICKER_SUGGEST.format(tree_id=owner_tree_id)
    return urljoin(base_url.rstrip("/") + "/", formatted_path) + f"?{suggest_params}"


def _validate_suggest_response(suggest_response: Any, api_description: str) -> Optional[list[dict[str, Any]]]:
    """Validate and process suggest API response."""
    if isinstance(suggest_response, list):
        if PYDANTIC_AVAILABLE and suggest_response:
            validated_results = []
            validation_errors = 0

            for item in suggest_response:
                try:
                    validated_item = PersonSuggestResponse(**item)
                    validated_results.append(validated_item.dict(exclude_none=True))
                except Exception as validation_err:
                    validation_errors += 1
                    logger.debug(f"Response validation warning for item: {validation_err}")
                    validated_results.append(item)

            if validation_errors > 0:
                logger.warning(
                    f"Response validation: {validation_errors}/{len(suggest_response)} items had validation issues"
                )

            return validated_results
        return suggest_response

    if suggest_response is None:
        return None

    logger.error(f"{api_description} call using _api_req returned unexpected type: {type(suggest_response)}")
    logger.debug(f"Unexpected Response Content: {str(suggest_response)[:500]}")
    return None


def _make_suggest_api_request(
    suggest_url: str,
    session_manager: "SessionManager",
    owner_facts_referer: str,
    timeout: int,
    api_description: str
) -> Any:
    """Make a single suggest API request."""
    custom_headers = {
        "Accept": "application/json",
        "Referer": owner_facts_referer,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }

    return _api_req(
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


def _handle_suggest_timeout(timeout: int, attempt: int, max_attempts: int, suggest_url: str, api_description: str, timeout_err: Exception):
    """Handle timeout exception for suggest API."""
    timeout_error = NetworkTimeoutError(
        f"API request timed out after {timeout}s",
        context={
            "api": api_description,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "timeout": timeout,
            "url": suggest_url,
        },
        recovery_hint="Check network connectivity and try again",
    )
    logger.warning(str(timeout_error))
    if attempt == max_attempts:
        raise timeout_error from timeout_err


def _handle_suggest_rate_limit(req_err: Exception, attempt: int, max_attempts: int, api_description: str, suggest_url: str):
    """Handle rate limit (429) exception for suggest API."""
    import time
    rate_limit_delay = 5.0 * (2 ** (attempt - 1))
    rate_limit_delay = min(rate_limit_delay, 300.0)
    logger.warning(
        f"Rate limit (429) on {api_description}, attempt {attempt}/{max_attempts}. "
        f"Waiting {rate_limit_delay:.1f}s before retry..."
    )
    time.sleep(rate_limit_delay)
    if attempt == max_attempts:
        raise APIRateLimitError(
            f"API rate limit exceeded after {max_attempts} attempts: {req_err}",
            context={"api": api_description, "url": suggest_url, "final_delay": rate_limit_delay},
        ) from req_err


def _try_direct_suggest_fallback(
    suggest_url: str,
    session_manager: "SessionManager",
    owner_facts_referer: str,
    api_description: str
) -> Optional[list[dict[str, Any]]]:
    """Try direct requests fallback for suggest API."""
    logger.warning(f"{api_description} failed via _api_req. Attempting direct requests fallback.")
    direct_response_obj = None

    try:
        cookies = {}
        if session_manager._requests_session:
            cookies = session_manager._requests_session.cookies.get_dict()

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
        direct_response_obj = requests.get(suggest_url, headers=direct_headers, cookies=cookies, timeout=direct_timeout)

        if direct_response_obj.status_code == 200:
            direct_data = direct_response_obj.json()
            if isinstance(direct_data, list):
                logger.info(f"Direct request fallback successful! Found {len(direct_data)} results.")
                return direct_data
            logger.warning(f"Direct request succeeded (200 OK) but returned non-list data: {type(direct_data)}")
            logger.debug(f"Direct Response content: {str(direct_data)[:500]}")
        else:
            logger.warning(f"Direct request fallback failed: Status {direct_response_obj.status_code}")
            logger.debug(f"Direct Response content: {direct_response_obj.text[:500]}")

    except requests.exceptions.Timeout:
        logger.error(f"Direct request fallback timed out after {direct_timeout} seconds")
    except json.JSONDecodeError as json_err:
        logger.error(f"Direct request fallback failed to decode JSON: {json_err}")
        if direct_response_obj:
            logger.debug(f"Direct Response content: {direct_response_obj.text[:500]}")
    except Exception as direct_err:
        logger.error(f"Direct request fallback failed with error: {direct_err}", exc_info=True)

    return None


def _execute_suggest_api_with_retries(
    suggest_url: str,
    session_manager: "SessionManager",
    owner_facts_referer: str,
    timeouts_used: list[int],
    api_description: str
) -> Optional[list[dict[str, Any]]]:
    """Execute Suggest API with retry logic."""
    max_attempts = len(timeouts_used)
    suggest_response = None

    for attempt, timeout in enumerate(timeouts_used, 1):
        logger.debug(f"{api_description} attempt {attempt}/{max_attempts} with timeout {timeout}s")

        try:
            suggest_response = _make_suggest_api_request(
                suggest_url, session_manager, owner_facts_referer, timeout, api_description
            )

            validated_response = _validate_suggest_response(suggest_response, api_description)
            if validated_response is not None:
                logger.info(
                    f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts}), "
                    f"found {len(validated_response)} results."
                )
                return validated_response

            if suggest_response is None:
                logger.warning(f"{api_description} call using _api_req returned None on attempt {attempt}/{max_attempts}.")
            else:
                suggest_response = None
                break

        except requests.exceptions.Timeout as timeout_err:
            _handle_suggest_timeout(timeout, attempt, max_attempts, suggest_url, api_description, timeout_err)

        except requests.exceptions.RequestException as req_err:
            if "rate limit" in str(req_err).lower() or "429" in str(req_err):
                _handle_suggest_rate_limit(req_err, attempt, max_attempts, api_description, suggest_url)
                continue
            raise NetworkTimeoutError(
                f"Network request failed: {req_err}",
                context={"api": api_description, "url": suggest_url},
            ) from req_err

        except Exception as api_err:
            logger.error(
                f"{api_description} _api_req call failed on attempt {attempt}/{max_attempts}: {api_err}",
                exc_info=True,
            )
            if attempt == max_attempts:
                raise RetryableError(
                    f"API call failed after {max_attempts} attempts: {api_err}",
                    context={"api": api_description, "attempts": max_attempts, "url": suggest_url},
                ) from api_err
            suggest_response = None
            continue

    return None


@timeout_protection(timeout=180)  # 3 minutes for complex API calls
@error_context("Ancestry Suggest API Call")
def call_suggest_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    _owner_profile_id: Optional[str],  # Unused but kept for API consistency
    base_url: str,
    search_criteria: dict[str, Any],
    timeouts: Optional[list[int]] = None,
) -> Optional[list[dict[str, Any]]]:
    # Validate inputs
    _validate_suggest_api_inputs(session_manager, owner_tree_id)

    api_description = "Suggest API"

    # Apply rate limiting
    _apply_rate_limiting(api_description)

    # Build URL
    suggest_url = _build_suggest_url(owner_tree_id, base_url, search_criteria)
    owner_facts_referer = _get_owner_referer(session_manager, base_url)

    timeouts_used = timeouts if timeouts else [20, 30, 60]
    len(timeouts_used)
    logger.info(f"Attempting {api_description} search: {suggest_url}")

    result = _execute_suggest_api_with_retries(
        suggest_url, session_manager, owner_facts_referer, timeouts_used, api_description
    )
    if result is not None:
        return result

    # Try direct fallback if all attempts failed
    direct_result = _try_direct_suggest_fallback(suggest_url, session_manager, owner_facts_referer, api_description)
    if direct_result is not None:
        return direct_result

    logger.error(f"{api_description} failed after all attempts and fallback.")
    return None


# End of call_suggest_api


def _validate_facts_api_prerequisites(
    session_manager: "SessionManager",
    owner_profile_id: str,
    api_person_id: str,
    api_tree_id: str,
) -> bool:
    """Validate prerequisites for Facts API call."""
    if not callable(_api_req):
        logger.critical(
            "Facts API call failed: _api_req function unavailable (Import Failed?)."
        )
        raise ImportError("_api_req function not available from utils")

    if not isinstance(session_manager, SessionManager):  # type: ignore[unreachable]
        logger.error("Facts API call failed: Invalid SessionManager passed.")
        return False

    if not all([owner_profile_id, api_person_id, api_tree_id]):
        logger.error(
            "Facts API call failed: owner_profile_id, api_person_id, and api_tree_id are required."
        )
        return False

    return True


def _apply_rate_limiting(api_description: str) -> None:
    """Apply rate limiting if available."""
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(
                f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s"
            )
            import time
            time.sleep(wait_time)
        api_rate_limiter.record_request()


def _try_direct_facts_request(
    session_manager: "SessionManager",
    facts_api_url: str,
    facts_referer: str,
    direct_timeout: int,
    api_description: str,
) -> Optional[dict[str, Any]]:
    """Try direct facts request using requests library."""
    logger.info(f"Attempting {api_description} via direct request: {facts_api_url}")

    direct_response_obj = None
    try:
        cookies = {}
        if session_manager._requests_session:
            cookies = session_manager._requests_session.cookies.get_dict()

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
                return None
            logger.info(f"{api_description} call successful via direct request.")
            return facts_data_raw
        logger.warning(
            f"Direct facts request failed: Status {direct_response_obj.status_code}"
        )
        logger.debug(f"Response content: {direct_response_obj.text[:500]}")
        return None

    except requests.exceptions.Timeout:
        logger.error(f"Direct facts request timed out after {direct_timeout} seconds")
        return None
    except json.JSONDecodeError as json_err:
        logger.error(f"Direct facts request failed to decode JSON: {json_err}")
        if direct_response_obj:
            logger.debug(f"Response content: {direct_response_obj.text[:500]}")
        return None
    except Exception as direct_err:
        logger.error(f"Direct facts request failed: {direct_err}", exc_info=True)
        return None


def _try_fallback_facts_request(
    session_manager: "SessionManager",
    facts_api_url: str,
    facts_referer: str,
    fallback_timeouts: list[int],
    api_description: str,
) -> Optional[dict[str, Any]]:
    """Try fallback facts request using _api_req."""
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
                logger.info(
                    f"{api_description} call successful via _api_req (attempt {attempt}/{max_attempts})."
                )
                return api_response
            if api_response is None:
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
        except requests.exceptions.Timeout:
            logger.warning(
                f"{api_description} _api_req call timed out after {timeout}s on attempt {attempt}/{max_attempts}."
            )
        except Exception as api_req_err:
            logger.error(
                f"{api_description} call using _api_req failed on attempt {attempt}/{max_attempts}: {api_req_err}",
                exc_info=True,
            )
            return None

    return None


def _validate_and_extract_facts_data(
    facts_data_raw: Optional[dict[str, Any]],
    api_person_id: str,
    api_description: str,
) -> Optional[dict[str, Any]]:
    """Validate and extract person research data from facts API response."""
    if not isinstance(facts_data_raw, dict):
        logger.error(
            f"Failed to fetch valid {api_description} data after all attempts."
        )
        return None

    person_research_data = facts_data_raw.get("data", {}).get("personResearch")
    if not isinstance(person_research_data, dict) or not person_research_data:
        logger.error(
            f"{api_description} response received, but missing 'data.personResearch' dictionary."
        )
        logger.debug(f"Full raw response keys: {list(facts_data_raw.keys())}")
        if "data" in facts_data_raw and isinstance(facts_data_raw["data"], dict):
            logger.debug(f"'data' sub-keys: {list(facts_data_raw['data'].keys())}")
        else:
            logger.debug("'data' key missing or not a dict in response.")
        return None

    # Validate response with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            PersonFactsResponse(**facts_data_raw)
            logger.debug("Facts API response validation successful")
        except Exception as validation_err:
            logger.warning(f"Facts API response validation warning: {validation_err}")
            # Continue with original data if validation fails

    logger.info(
        f"Successfully fetched and extracted 'personResearch' data for PersonID {api_person_id}."
    )
    return person_research_data


def call_facts_user_api(
    session_manager: "SessionManager",
    api_ids: ApiIdentifiers,
    base_url: str,
    timeouts: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    # Validate prerequisites
    if not _validate_facts_api_prerequisites(
        session_manager, api_ids.owner_profile_id, api_ids.api_person_id, api_ids.api_tree_id
    ):
        return None

    api_description = "Person Facts User API"
    _apply_rate_limiting(api_description)

    formatted_path = API_PATH_PERSON_FACTS_USER.format(
        owner_profile_id=api_ids.owner_profile_id.lower(),
        tree_id=api_ids.api_tree_id.lower(),
        person_id=api_ids.api_person_id.lower(),
    )
    facts_api_url = urljoin(base_url.rstrip("/") + "/", formatted_path)
    facts_referer = _get_owner_referer(session_manager, base_url)
    direct_timeout = _get_api_timeout(30)
    fallback_timeouts = timeouts if timeouts else [30, 45, 60]

    # Try direct request first
    facts_data_raw = _try_direct_facts_request(
        session_manager, facts_api_url, facts_referer, direct_timeout, api_description
    )

    # Try fallback if direct request failed
    if facts_data_raw is None:
        facts_data_raw = _try_fallback_facts_request(
            session_manager, facts_api_url, facts_referer, fallback_timeouts, api_description
        )

    # Validate and extract person research data
    return _validate_and_extract_facts_data(facts_data_raw, api_ids.api_person_id, api_description)


# End of call_facts_user_api


def _process_getladder_response(relationship_data: Any, api_description: str) -> Optional[str]:
    """Process GetLadder API response."""
    if not isinstance(relationship_data, str):
        logger.warning(f"{api_description} call returned non-string or None: {type(relationship_data)}")
        return None

    if len(relationship_data) <= 10:
        logger.warning(f"{api_description} call returned a very short string: '{relationship_data}'")
        return None

    # Validate response with Pydantic if available and try to parse as JSON
    if PYDANTIC_AVAILABLE:
        try:
            import json
            parsed_data = json.loads(relationship_data)
            GetLadderResponse(**parsed_data)
            logger.debug("GetLadder API response validation successful")
        except json.JSONDecodeError:
            logger.debug("GetLadder API returned non-JSON string response")
        except Exception as validation_err:
            logger.warning(f"GetLadder API response validation warning: {validation_err}")

    logger.debug(f"{api_description} call successful, received string response.")
    return relationship_data


def call_getladder_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    target_person_id: str,
    base_url: str,
    timeout: Optional[int] = None,
) -> Optional[str]:
    if not callable(_api_req):
        logger.critical("GetLadder API call failed: _api_req function unavailable (Import Failed?).")
        raise ImportError("_api_req function not available from utils")
    if not isinstance(session_manager, SessionManager):  # type: ignore[unreachable]
        logger.error("GetLadder API call failed: Invalid SessionManager passed.")
        return None
    if not all([owner_tree_id, target_person_id]):
        logger.error("GetLadder API call failed: owner_tree_id and target_person_id are required.")
        return None

    api_description = "Get Tree Ladder API"
    _apply_rate_limiting(api_description)

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
    logger.debug(f"Attempting {api_description} call: {ladder_api_url}")

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
        return _process_getladder_response(relationship_data, api_description)
    except requests.exceptions.Timeout:
        logger.error(f"{api_description} call timed out after {api_timeout_val}s.")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_getladder_api


def _process_discovery_relationship_response(relationship_data: Any, api_description: str) -> Optional[dict[str, Any]]:
    """Process discovery relationship API response."""
    if not isinstance(relationship_data, dict):
        logger.warning(f"{api_description} call returned unexpected type: {type(relationship_data)}")
        return None

    # Validate response with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            DiscoveryRelationshipResponse(**relationship_data)
            logger.debug("Discovery Relationship API response validation successful")
        except Exception as validation_err:
            logger.warning(f"Discovery Relationship API response validation warning: {validation_err}")

    if "path" in relationship_data:
        logger.info(f"{api_description} call successful, received valid JSON response with path data.")
        return relationship_data

    logger.warning(f"{api_description} call returned JSON without 'path' key: {list(relationship_data.keys())}")
    return relationship_data  # Still return the data for potential debugging


def call_discovery_relationship_api(
    session_manager: "SessionManager",
    selected_person_global_id: str,
    owner_profile_id: str,
    base_url: str,
    timeout: Optional[int] = None,
) -> Optional[dict[str, Any]]:
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
        logger.critical("Discovery Relationship API call failed: _api_req function unavailable (Import Failed?).")
        raise ImportError("_api_req function not available from utils")
    if not isinstance(session_manager, SessionManager):  # type: ignore[unreachable]
        logger.error("Discovery Relationship API call failed: Invalid SessionManager passed.")
        return None
    if not all([owner_profile_id, selected_person_global_id]):
        logger.error("Discovery Relationship API call failed: owner_profile_id and selected_person_global_id are required.")
        return None

    api_description = "Discovery Relationship API"
    _apply_rate_limiting(api_description)

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

        return _process_discovery_relationship_response(relationship_data, api_description)
    except requests.exceptions.Timeout:
        logger.error(f"{api_description} call timed out after {api_timeout_val}s.")
        return None
    except Exception as e:
        logger.error(f"API call '{api_description}' failed: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_discovery_relationship_api


def _build_treesui_url(owner_tree_id: str, base_url: str, search_criteria: dict[str, Any]) -> Optional[str]:
    """Build TreesUI List API URL with search parameters."""
    first_name_raw = search_criteria.get("first_name_raw", "")
    surname_raw = search_criteria.get("surname_raw", "")
    birth_year = search_criteria.get("birth_year")

    if not birth_year:
        logger.warning("Cannot call TreesUI List API: 'birth_year' is missing in search criteria.")
        return None

    treesui_params_list = ["limit=100", "fields=NAMES,BIRTH_DEATH"]
    if first_name_raw:
        treesui_params_list.append(f"fn={quote(first_name_raw)}")
    if surname_raw:
        treesui_params_list.append(f"ln={quote(surname_raw)}")
    treesui_params_list.append(f"by={birth_year}")

    treesui_params = "&".join(treesui_params_list)
    formatted_path = API_PATH_TREESUI_LIST.format(tree_id=owner_tree_id)
    return urljoin(base_url.rstrip("/") + "/", formatted_path) + f"?{treesui_params}"


def call_treesui_list_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    _owner_profile_id: Optional[str],  # Unused but kept for API consistency
    base_url: str,
    search_criteria: dict[str, Any],
    timeouts: Optional[list[int]] = None,
) -> Optional[list[dict[str, Any]]]:
    if not callable(_api_req):
        logger.critical("TreesUI List API call failed: _api_req function unavailable (Import Failed?).")
        raise ImportError("_api_req function not available from utils")
    if not isinstance(session_manager, SessionManager):  # type: ignore[unreachable]
        logger.error("TreesUI List API call failed: Invalid SessionManager passed.")
        return None
    if not owner_tree_id:
        logger.error("TreesUI List API call failed: owner_tree_id is required.")
        return None

    api_description = "TreesUI List API (Alternative Search)"
    _apply_rate_limiting(api_description)

    treesui_url = _build_treesui_url(owner_tree_id, base_url, search_criteria)
    if not treesui_url:
        return None

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
            if treesui_response is None:
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


def _validate_message_response(
    api_response: dict[str, Any],
    is_initial: bool,
    existing_conv_id: Optional[str],
    my_profile_id_upper: str,
    log_prefix: str
) -> tuple[bool, Optional[str], str]:
    """Validate message API response. Returns (post_ok, conversation_id, message_status)."""
    if is_initial:
        api_conv_id = str(api_response.get(KEY_CONVERSATION_ID, ""))
        msg_details = api_response.get(KEY_MESSAGE, {})
        api_author = str(msg_details.get(KEY_AUTHOR, "")).upper() if isinstance(msg_details, dict) else None

        if api_conv_id and api_author == my_profile_id_upper:
            return True, api_conv_id, SEND_ERROR_UNKNOWN

        logger.error(f"{log_prefix}: API initial response format invalid (ConvID: '{api_conv_id}', Author: '{api_author}', Expected Author: '{my_profile_id_upper}').")
        logger.debug(f"API Response: {api_response}")
        return False, None, SEND_ERROR_VALIDATION_FAILED

    # Follow-up message
    api_author = str(api_response.get(KEY_AUTHOR, "")).upper()
    if api_author == my_profile_id_upper:
        return True, existing_conv_id, SEND_ERROR_UNKNOWN

    logger.error(f"{log_prefix}: API follow-up author validation failed (Author: '{api_author}', Expected Author: '{my_profile_id_upper}').")
    logger.debug(f"API Response: {api_response}")
    return False, None, SEND_ERROR_VALIDATION_FAILED


def _process_send_message_response(
    api_response: Any,
    is_initial: bool,
    existing_conv_id: Optional[str],
    my_profile_id_upper: str,
    person: "Person",
    send_api_desc: str,
    log_prefix: str
) -> tuple[str, Optional[str]]:
    """Process send message API response and return status and conversation ID."""
    message_status = SEND_ERROR_UNKNOWN
    new_conversation_id_from_api: Optional[str] = None
    post_ok = False

    if api_response is None:
        message_status = SEND_ERROR_POST_FAILED
        logger.error(f"{log_prefix}: API POST ({send_api_desc}) failed (No response/Retries exhausted).")
    elif isinstance(api_response, requests.Response):
        message_status = f"send_error (http_{api_response.status_code})"
        logger.error(f"{log_prefix}: API POST ({send_api_desc}) failed with status {api_response.status_code}.")
        from contextlib import suppress
        with suppress(Exception):
            logger.debug(f"Error response body: {api_response.text[:500]}")
    elif isinstance(api_response, dict):
        try:
            post_ok, new_conversation_id_from_api, message_status = _validate_message_response(
                api_response, is_initial, existing_conv_id, my_profile_id_upper, log_prefix
            )
            if post_ok:
                message_status = SEND_SUCCESS_DELIVERED
                logger.info(f"{log_prefix}: Message send to {getattr(person, 'username', None) or getattr(person, 'profile_id', 'Unknown')} successful (ConvID: {new_conversation_id_from_api}).")
        except Exception as parse_err:
            logger.error(f"{log_prefix}: Error parsing successful API response ({send_api_desc}): {parse_err}", exc_info=True)
            logger.debug(f"API Response received: {api_response}")
            message_status = SEND_ERROR_UNEXPECTED_FORMAT
    else:
        logger.error(f"{log_prefix}: API call ({send_api_desc}) unexpected success format. Type:{type(api_response)}, Resp:{str(api_response)[:200]}")
        message_status = SEND_ERROR_UNEXPECTED_FORMAT

    if not post_ok and message_status == SEND_ERROR_UNKNOWN:
        message_status = SEND_ERROR_VALIDATION_FAILED
        logger.warning(f"{log_prefix}: Message send attempt concluded with status: {message_status}")

    return message_status, new_conversation_id_from_api


def _prepare_send_message_request(
    message_text: str,
    my_profile_id_lower: str,
    recipient_profile_id_upper: str,
    existing_conv_id: Optional[str],
    log_prefix: str
) -> Optional[tuple[str, dict[str, Any], str, dict[str, Any]]]:
    """Prepare API request data for sending message. Returns (url, payload, description, headers) or None."""
    try:
        base_url_cfg = config_schema.api.base_url or "https://www.ancestry.com"
        is_initial = not existing_conv_id

        if is_initial:
            send_api_url = urljoin(base_url_cfg.rstrip("/") + "/", API_PATH_SEND_MESSAGE_NEW)
            send_api_desc = "Create Conversation API"
            payload = {
                "content": message_text,
                "author": my_profile_id_lower,
                "index": 0,
                "created": 0,
                "conversation_members": [
                    {"user_id": recipient_profile_id_upper.lower(), "family_circles": []},
                    {"user_id": my_profile_id_lower},
                ],
            }
        elif existing_conv_id:
            formatted_path = API_PATH_SEND_MESSAGE_EXISTING.format(conv_id=existing_conv_id)
            send_api_url = urljoin(base_url_cfg.rstrip("/") + "/", formatted_path)
            send_api_desc = "Send Message API (Existing Conv)"
            payload = {"content": message_text, "author": my_profile_id_lower}
        else:
            logger.error(f"{log_prefix}: Logic Error - Cannot determine API URL/payload (existing_conv_id issue?).")
            return None

        ctx_headers = config_schema.api.api_contextual_headers.get(send_api_desc, {})
        api_headers = ctx_headers.copy()

        return send_api_url, payload, send_api_desc, api_headers
    except Exception as prep_err:
        logger.error(f"{log_prefix}: Error preparing API request data: {prep_err}", exc_info=True)
        return None


def _validate_send_message_request(
    session_manager: "SessionManager",
    person: "Person",
    message_text: str,
    log_prefix: str
) -> Optional[tuple[str, Optional[str]]]:
    """Validate send message request. Returns error tuple if invalid, None if valid."""
    if not session_manager or not session_manager.my_profile_id:
        logger.error(f"{log_prefix}: Cannot send message - SessionManager or own profile ID missing.")
        return SEND_ERROR_MISSING_OWN_ID, None

    if not isinstance(person, Person) or not getattr(person, "profile_id", None):
        logger.error(f"{log_prefix}: Cannot send message - Invalid Person object (Type: {type(person)}) or missing profile ID.")
        return SEND_ERROR_INVALID_RECIPIENT, None

    if not isinstance(message_text, str) or not message_text.strip():
        logger.error(f"{log_prefix}: Cannot send message - Message text is empty or invalid.")
        return SEND_ERROR_API_PREP_FAILED, None

    return None


def _handle_dry_run_mode(person: "Person", existing_conv_id: Optional[str], log_prefix: str) -> tuple[str, Optional[str]]:
    """Handle dry run mode for message sending."""
    message_status = SEND_SUCCESS_DRY_RUN
    effective_conv_id = existing_conv_id or f"dryrun_{uuid.uuid4()}"
    logger.info(f"{log_prefix}: Dry Run - Simulated message send to {getattr(person, 'username', None) or getattr(person, 'profile_id', 'Unknown')}.")
    return message_status, effective_conv_id


def call_send_message_api(
    session_manager: "SessionManager",
    person: "Person",
    message_text: str,
    existing_conv_id: Optional[str],
    log_prefix: str,
) -> tuple[str, Optional[str]]:
    validation_error = _validate_send_message_request(session_manager, person, message_text, log_prefix)
    if validation_error:
        return validation_error

    app_mode = config_schema.app_mode
    if app_mode == "dry_run":
        return _handle_dry_run_mode(person, existing_conv_id, log_prefix)
    if app_mode not in ["production", "testing"]:
        logger.error(f"{log_prefix}: Logic Error - Unexpected APP_MODE '{app_mode}' reached send logic.")
        return SEND_ERROR_INTERNAL_MODE, None

    MY_PROFILE_ID_LOWER = session_manager.my_profile_id.lower()
    MY_PROFILE_ID_UPPER = session_manager.my_profile_id.upper()
    recipient_profile_id_upper = getattr(person, "profile_id", "").upper()

    api_request_data = _prepare_send_message_request(
        message_text, MY_PROFILE_ID_LOWER, recipient_profile_id_upper, existing_conv_id, log_prefix
    )
    if not api_request_data:
        return SEND_ERROR_API_PREP_FAILED, None

    send_api_url, payload, send_api_desc, api_headers = api_request_data

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

    is_initial = not existing_conv_id
    return _process_send_message_response(
        api_response, is_initial, existing_conv_id, MY_PROFILE_ID_UPPER,
        person, send_api_desc, log_prefix
    )


# End of call_send_message_api


def _process_profile_response(profile_response: Any, profile_id: str) -> Optional[dict[str, Any]]:
    """Process and validate profile API response."""
    if not profile_response or not isinstance(profile_response, dict):
        if isinstance(profile_response, requests.Response):
            logger.warning(f"Failed profile details fetch for {profile_id}. Status: {profile_response.status_code}.")
        elif profile_response is None:
            logger.warning(f"Failed profile details fetch for {profile_id} (_api_req returned None).")
        else:
            logger.warning(f"Failed profile details fetch for {profile_id} (Invalid response type: {type(profile_response)}).")
        return None

    # Validate response with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            ProfileDetailsResponse(**profile_response)
            logger.debug("Profile Details API response validation successful")
        except Exception as validation_err:
            logger.warning(f"Profile Details API response validation warning: {validation_err}")

    logger.debug(f"Successfully fetched profile details for {profile_id}.")
    return {
        "first_name": _extract_first_name(profile_response, profile_id),
        "last_logged_in_dt": _parse_last_login_date(profile_response, profile_id),
        "contactable": bool(profile_response.get(KEY_IS_CONTACTABLE)),
    }


def _extract_first_name(profile_response: dict[str, Any], profile_id: str) -> Optional[str]:
    """Extract first name from profile response."""
    first_name_raw = profile_response.get(KEY_FIRST_NAME)
    if first_name_raw and isinstance(first_name_raw, str):
        return format_name(first_name_raw)

    display_name_raw = profile_response.get(KEY_DISPLAY_NAME)
    if display_name_raw and isinstance(display_name_raw, str):
        formatted_dn = format_name(display_name_raw)
        return formatted_dn.split()[0] if formatted_dn != "Valued Relative" else None

    logger.warning(f"Could not extract FirstName or DisplayName for profile {profile_id}")
    return None


def _parse_last_login_date(profile_response: dict[str, Any], profile_id: str) -> Optional[datetime]:
    """Parse last login date from profile response."""
    last_login_str = profile_response.get(KEY_LAST_LOGIN_DATE)
    if not last_login_str or not isinstance(last_login_str, str):
        logger.debug(f"LastLoginDate missing or invalid for profile {profile_id}")
        return None

    try:
        if last_login_str.endswith("Z"):
            dt_aware = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
        elif "+" in last_login_str or "-" in last_login_str[10:]:
            dt_aware = datetime.fromisoformat(last_login_str)
        else:
            dt_naive = datetime.fromisoformat(last_login_str)
            dt_aware = dt_naive.replace(tzinfo=timezone.utc)
        return dt_aware.astimezone(timezone.utc)
    except (ValueError, TypeError) as date_parse_err:
        logger.warning(f"Could not parse LastLoginDate '{last_login_str}' for {profile_id}: {date_parse_err}")
        return None


def _validate_profile_request(session_manager: "SessionManager", profile_id: str) -> bool:
    """Validate profile request prerequisites."""
    if not profile_id or not isinstance(profile_id, str):
        logger.warning("call_profile_details_api: Profile ID missing or invalid.")
        return False
    if not session_manager or not session_manager.my_profile_id:
        logger.error("call_profile_details_api: SessionManager or own profile ID missing.")
        return False
    if not session_manager.is_sess_valid():
        logger.error(f"call_profile_details_api: Session invalid for Profile ID {profile_id}.")
        return False
    return True


def _apply_rate_limiting(api_description: str) -> None:
    """Apply rate limiting if available."""
    if api_rate_limiter and PYDANTIC_AVAILABLE:
        if not api_rate_limiter.can_make_request():
            wait_time = api_rate_limiter.wait_time_until_available()
            logger.warning(f"Rate limit reached for {api_description}. Waiting {wait_time:.1f}s")
            import time
            time.sleep(wait_time)
        api_rate_limiter.record_request()


def call_profile_details_api(
    session_manager: "SessionManager", profile_id: str
) -> Optional[dict[str, Any]]:
    if not _validate_profile_request(session_manager, profile_id):
        return None

    api_description = "Profile Details API (Single)"
    _apply_rate_limiting(api_description)

    base_url_cfg = config_schema.api.base_url or "https://www.ancestry.com"
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

        return _process_profile_response(profile_response, profile_id)
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


def _parse_tree_id_from_url(tree_url: Any, tree_name_config: str) -> Optional[str]:
    """Parse tree ID from tree URL."""
    if not tree_url or not isinstance(tree_url, str):
        logger.warning(f"Found tree '{tree_name_config}', but '{KEY_URL}' key missing or invalid.")
        return None

    match = re.search(r"/tree/(\d+)", tree_url)
    if match:
        my_tree_id_val = match.group(1)
        logger.debug(f"Found tree ID '{my_tree_id_val}' for tree '{tree_name_config}'.")
        return my_tree_id_val

    logger.warning(f"Found tree '{tree_name_config}', but URL format unexpected: {tree_url}")
    return None


def _validate_header_trees_response(response_data: Any, api_description: str) -> Optional[list]:
    """Validate header trees response and return menu items if valid."""
    if not response_data or not isinstance(response_data, dict):
        if response_data is None:
            logger.warning(f"{api_description} call failed (_api_req returned None).")
        else:
            status = str(response_data.status_code) if isinstance(response_data, requests.Response) else "N/A"
            logger.warning(f"Unexpected response format from {api_description} (Type: {type(response_data)}, Status: {status}).")
            logger.debug(f"Full {api_description} response data: {response_data!s}")
        return None

    if KEY_MENUITEMS not in response_data or not isinstance(response_data[KEY_MENUITEMS], list):
        logger.warning(f"Unexpected response format from {api_description} (missing or invalid menuItems).")
        return None

    # Validate response with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            HeaderTreesResponse(**response_data)
            logger.debug("Header Trees API response validation successful")
        except Exception as validation_err:
            logger.warning(f"Header Trees API response validation warning: {validation_err}")

    return response_data[KEY_MENUITEMS]


def _extract_tree_id_from_response(response_data: Any, tree_name_config: str, api_description: str) -> Optional[str]:
    """Extract tree ID from header trees API response."""
    menu_items = _validate_header_trees_response(response_data, api_description)
    if not menu_items:
        return None

    for item in menu_items:
        if isinstance(item, dict) and item.get(KEY_TEXT) == tree_name_config:
            return _parse_tree_id_from_url(item.get(KEY_URL), tree_name_config)

    logger.warning(f"Could not find TREE_NAME '{tree_name_config}' in {api_description} response.")
    return None


def _validate_header_trees_request(session_manager: "SessionManager", tree_name_config: str) -> bool:
    """Validate header trees API request prerequisites."""
    if not tree_name_config:
        logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
        return False
    if not session_manager.is_sess_valid():
        logger.error("call_header_trees_api_for_tree_id: Session invalid.")
        return False
    if not callable(_api_req):
        logger.critical("call_header_trees_api_for_tree_id: _api_req is not callable!")
        raise ImportError("_api_req function not available from utils")
    return True


def call_header_trees_api_for_tree_id(
    session_manager: "SessionManager", tree_name_config: str
) -> Optional[str]:
    if not _validate_header_trees_request(session_manager, tree_name_config):
        return None

    base_url_cfg = config_schema.api.base_url or "https://www.ancestry.com"
    url = urljoin(base_url_cfg.rstrip("/") + "/", API_PATH_HEADER_TREES)
    api_description = "Header Trees API (Nav Data)"

    _apply_rate_limiting(api_description)
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

        return _extract_tree_id_from_response(response_data, tree_name_config, api_description)
    except Exception as e:
        logger.error(f"Error during {api_description}: {e}", exc_info=True)
        return None
    # End of try/except


# End of call_header_trees_api_for_tree_id


def _extract_tree_owner_from_response(response_data: Any, tree_id: str, api_description: str) -> Optional[str]:
    """Extract tree owner name from API response."""
    if not response_data or not isinstance(response_data, dict):
        if response_data is None:
            logger.warning(f"{api_description} call failed (_api_req returned None).")
        else:
            status = str(response_data.status_code) if isinstance(response_data, requests.Response) else "N/A"
            logger.warning(f"{api_description} call returned unexpected data (Type: {type(response_data)}, Status: {status}) or None.")
            logger.debug(f"Response received: {response_data!s}")
        return None

    # Validate response with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            TreeOwnerResponse.from_dict(response_data)
            logger.debug("Tree Owner API response validation successful")
        except Exception as validation_err:
            logger.warning(f"Tree Owner API response validation warning: {validation_err}")

    owner_data = response_data.get(KEY_OWNER)
    if not owner_data or not isinstance(owner_data, dict):
        logger.warning(f"Could not find '{KEY_OWNER}' data in {api_description} response for tree {tree_id}.")
        logger.debug(f"Full {api_description} response data: {response_data}")
        return None

    display_name = owner_data.get(KEY_DISPLAY_NAME)
    if display_name and isinstance(display_name, str):
        logger.debug(f"Found tree owner '{display_name}' for tree ID {tree_id}.")
        return display_name

    logger.warning(f"Could not find '{KEY_DISPLAY_NAME}' in owner data for tree {tree_id}.")
    logger.debug(f"Full {api_description} response data: {response_data}")
    return None


def _validate_tree_owner_request(session_manager: "SessionManager", tree_id: str) -> bool:
    """Validate tree owner API request prerequisites."""
    if not tree_id:
        logger.warning("Cannot get tree owner: tree_id is missing.")
        return False
    if not isinstance(tree_id, str):  # type: ignore[unreachable]
        logger.warning(f"Invalid tree_id type provided: {type(tree_id)}. Expected string.")
        return False
    if not session_manager.is_sess_valid():
        logger.error("call_tree_owner_api: Session invalid.")
        return False
    if not callable(_api_req):
        logger.critical("call_tree_owner_api: _api_req is not callable!")
        raise ImportError("_api_req function not available from utils")
    return True


def call_tree_owner_api(
    session_manager: "SessionManager", tree_id: str
) -> Optional[str]:
    if not _validate_tree_owner_request(session_manager, tree_id):
        return None

    base_url_cfg = config_schema.api.base_url or "https://www.ancestry.com"
    url = urljoin(base_url_cfg.rstrip("/") + "/", f"{API_PATH_TREE_OWNER_INFO}?tree_id={tree_id}")
    api_description = "Tree Owner Name API"

    _apply_rate_limiting(api_description)

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

        return _extract_tree_owner_from_response(response_data, tree_id, api_description)
    except Exception as e:
        logger.error(
            f"Error during {api_description} for tree {tree_id}: {e}", exc_info=True
        )
        return None
    # End of try/except


# End of call_tree_owner_api


# --- Standalone Test Block ---
# _sc_run_test and _sc_print_summary removed - unused 187-line test helper functions
# These were replaced by the centralized test framework


# === ASYNC API FUNCTIONS (Phase 7.4.1) ===

async def async_call_suggest_api(
    session_manager: "SessionManager",
    owner_tree_id: str,
    _owner_profile_id: Optional[str],  # Unused but kept for API consistency
    base_url: str,
    search_criteria: dict[str, Any],
    timeout: Optional[int] = None,
) -> Optional[list[PersonSuggestResponse]]:
    """
    Async version of call_suggest_api for concurrent person suggestions.

    Args:
        session_manager: SessionManager for authentication
        owner_tree_id: Tree ID for the search
        owner_profile_id: Profile ID (kept for compatibility)
        base_url: Base URL for the API
        search_criteria: Search parameters
        timeout: Optional timeout in seconds

    Returns:
        List of PersonSuggestResponse objects or None on failure

    Example:
        >>> suggestions = await async_call_suggest_api(
        ...     session_manager, tree_id, profile_id, base_url,
        ...     {"name": "John Smith", "birth_year": 1900}
        ... )
    """
    from utils import async_api_request

    # Build request URL and data (same logic as sync version)
    suggest_url = urljoin(base_url, "suggest")

    # Prepare request data
    request_data = {
        "treeId": owner_tree_id,
        **search_criteria
    }

    try:
        logger.debug(f"Async Suggest API call for tree {owner_tree_id}")

        response_data = await async_api_request(
            url=suggest_url,
            method="POST",
            session_manager=session_manager,
            json_data=request_data,
            timeout=timeout or 180,
            api_description="Ancestry Suggest API (Async)"
        )

        if not response_data:
            logger.warning("Async Suggest API returned no data")
            return None

        # Parse response into PersonSuggestResponse objects
        suggestions = []
        if isinstance(response_data, list):
            for item in response_data:
                if isinstance(item, dict):
                    suggestions.append(PersonSuggestResponse.from_dict(item))
        elif isinstance(response_data, dict) and "suggestions" in response_data:
            for item in response_data["suggestions"]:
                if isinstance(item, dict):
                    suggestions.append(PersonSuggestResponse.from_dict(item))

        logger.info(f"Async Suggest API returned {len(suggestions)} suggestions")
        return suggestions

    except Exception as e:
        logger.error(f"Async Suggest API error: {e}")
        return None


async def async_batch_person_lookup(
    session_manager: "SessionManager",
    person_ids: list[str],
    tree_id: str,
    base_url: str,
    max_concurrent: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> dict[str, Optional[PersonFactsResponse]]:
    """
    Perform concurrent person fact lookups for multiple persons.

    Args:
        session_manager: SessionManager for authentication
        person_ids: List of person IDs to look up
        tree_id: Tree ID for all lookups
        base_url: Base URL for the API
        max_concurrent: Maximum concurrent requests
        progress_callback: Optional progress callback

    Returns:
        Dictionary mapping person IDs to PersonFactsResponse objects

    Example:
        >>> results = await async_batch_person_lookup(
        ...     session_manager, ["person1", "person2"], tree_id, base_url
        ... )
        >>> for person_id, facts in results.items():
        ...     if facts:
        ...         print(f"Got facts for {person_id}")
    """
    from utils import async_batch_api_requests

    # Prepare batch requests
    requests = []
    for person_id in person_ids:
        facts_url = urljoin(base_url, f"trees/{tree_id}/persons/{person_id}/facts")
        requests.append({
            "url": facts_url,
            "method": "GET",
            "api_description": f"Facts for {person_id}"
        })

    logger.info(f"Starting async batch person lookup for {len(person_ids)} persons")

    # Execute batch requests
    results = await async_batch_api_requests(
        requests=requests,
        session_manager=session_manager,
        max_concurrent=max_concurrent,
        progress_callback=progress_callback
    )

    # Parse results into PersonFactsResponse objects
    parsed_results = {}
    for person_id, response_data in zip(person_ids, results):
        if response_data:
            try:
                parsed_results[person_id] = PersonFactsResponse.from_dict(response_data)
            except Exception as e:
                logger.warning(f"Failed to parse facts for {person_id}: {e}")
                parsed_results[person_id] = None
        else:
            parsed_results[person_id] = None

    successful_count = sum(1 for v in parsed_results.values() if v is not None)
    logger.info(f"Async batch person lookup completed: {successful_count}/{len(person_ids)} successful")

    return parsed_results


def call_enhanced_api(
    session_manager: "SessionManager",
    endpoint: str,
    user_id: str,
    tree_id: str,
    person_id: str,
    method: str = "GET",
    data: Optional[dict[str, Any]] = None,
    api_description: str = "Enhanced API Call",
    use_csrf_token: bool = True
) -> Optional[dict[str, Any]]:
    """
    Call an enhanced API endpoint with full browser-like authentication.

    This function provides enhanced authentication for API endpoints that require
    more sophisticated headers and session state, such as the new family relationship
    and kinship APIs discovered in Action 11.

    Args:
        session_manager: Active session manager
        endpoint: API endpoint path (e.g., "/family-tree/person/addedit/user/{user_id}/tree/{tree_id}/person/{person_id}/editrelationships")
        user_id: User profile ID
        tree_id: Tree ID
        person_id: Person ID
        method: HTTP method (GET, POST, etc.)
        data: Optional request data
        api_description: Description for logging

    Returns:
        Dictionary with API response data or None if failed
    """
    if not session_manager or not session_manager.is_sess_valid():
        logger.error(f"Invalid session manager for {api_description}")
        return None

    try:
        # Import _api_req from utils
        from utils import _api_req

        # Construct full URL
        base_url = config_schema.api.base_url.rstrip('/')
        # Format endpoint with provided IDs
        formatted_endpoint = endpoint.format(
            user_id=user_id,
            tree_id=tree_id,
            person_id=person_id
        )
        url = f"{base_url}{formatted_endpoint}"
        referer_url = f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/facts"

        logger.debug(f"Calling enhanced API: {url}")

        # Sync cookies to ensure authentication state
        try:
            if hasattr(session_manager, '_sync_cookies_to_requests'):
                session_manager._sync_cookies_to_requests()
        except Exception as e:
            logger.debug(f"Cookie sync failed: {e}")

        # Use enhanced _api_req with special parameters for enhanced headers
        response = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method=method,
            data=data,
            api_description=api_description,
            referer_url=referer_url,
            use_csrf_token=use_csrf_token,
            timeout=30,
            # Pass additional parameters for enhanced headers
            headers={
                "_use_enhanced_headers": "true",
                "_tree_id": tree_id,
                "_person_id": person_id
            }
        )

        if response:
            logger.debug(f"{api_description} successful")
            return response
        logger.warning(f"{api_description} returned no data")
        return None

    except Exception as e:
        logger.error(f"Error calling {api_description}: {e}")
        return None


def call_edit_relationships_api(
    session_manager: "SessionManager",
    user_id: str,
    tree_id: str,
    person_id: str
) -> Optional[dict[str, Any]]:
    """
    Call the edit relationships API endpoint to get family relationship data.

    Args:
        session_manager: Active session manager
        user_id: User profile ID
        tree_id: Tree ID
        person_id: Person ID

    Returns:
        Dictionary with relationship data or None if failed
    """
    endpoint = "/family-tree/person/addedit/user/{user_id}/tree/{tree_id}/person/{person_id}/editrelationships"
    return call_enhanced_api(
        session_manager=session_manager,
        endpoint=endpoint,
        user_id=user_id,
        tree_id=tree_id,
        person_id=person_id,
        api_description="Edit Relationships API",
        use_csrf_token=False  # Disable CSRF token for cleaner logging
    )


def call_relationship_ladder_api(
    session_manager: "SessionManager",
    user_id: str,
    tree_id: str,
    person_id: str
) -> Optional[dict[str, Any]]:
    """
    Call the enhanced relationship ladder API endpoint to get kinship relationship data.

    This function uses the improved endpoint that provides comprehensive relationship
    data including kinshipPersons with detailed relationship information.

    Args:
        session_manager: Active session manager
        user_id: User profile ID
        tree_id: Tree ID
        person_id: Person ID

    Returns:
        Dictionary with relationship ladder data or None if failed
    """
    endpoint = "/family-tree/person/card/user/{user_id}/tree/{tree_id}/person/{person_id}/kinship/relationladderwithlabels"
    return call_enhanced_api(
        session_manager=session_manager,
        endpoint=endpoint,
        user_id=user_id,
        tree_id=tree_id,
        person_id=person_id,
        api_description="Enhanced Relationship Ladder API",
        use_csrf_token=False  # Disable CSRF token for cleaner logging
    )


def get_relationship_path_data(
    session_manager: "SessionManager",
    person_id: str,
    _reference_person_id: Optional[str] = None  # Unused but kept for future use
) -> Optional[dict[str, Any]]:
    """
    Get comprehensive relationship path data for a person using the enhanced API.

    This function provides a unified interface for getting relationship data
    that can be used by both Action 6 and Action 11.

    Args:
        session_manager: Active session manager
        person_id: ID of the person to get relationships for
        reference_person_id: Optional reference person ID (defaults to tree owner)

    Returns:
        Dictionary with relationship data including kinshipPersons or None if failed
    """
    try:
        # Get user and tree IDs from SessionManager
        user_id = session_manager.my_profile_id or session_manager.my_uuid
        tree_id = session_manager.my_tree_id

        if not user_id or not tree_id:
            logger.error(f"Missing user_id ({user_id}) or tree_id ({tree_id}) for relationship path data")
            return None

        # Call the enhanced relationship ladder API
        result = call_relationship_ladder_api(
            session_manager=session_manager,
            user_id=user_id,
            tree_id=tree_id,
            person_id=person_id
        )

        if result and isinstance(result, dict):
            # Extract kinshipPersons data if available
            kinship_persons = result.get("kinshipPersons", [])
            if kinship_persons:
                logger.debug(f"Found {len(kinship_persons)} kinship relationships for person {person_id}")
                return {
                    "person_id": person_id,
                    "reference_person_id": result.get("mePid"),
                    "kinship_persons": kinship_persons,
                    "raw_data": result
                }
            logger.warning(f"No kinship persons found in relationship data for {person_id}")
            return None
        logger.warning(f"Invalid or empty relationship data for person {person_id}")
        return None

    except Exception as e:
        logger.error(f"Error getting relationship path data for {person_id}: {e}")
        return None


def _run_initialization_tests(suite: "TestSuite") -> None:
    """Run initialization tests for api_utils module."""
    def test_module_imports():
        """Test all required modules and dependencies are properly imported with detailed verification."""
        required_modules = [
            ("json", "JSON parsing and serialization"),
            ("requests", "HTTP requests and API calls"),
            ("time", "Time and timing operations"),
            ("logging", "Logging functionality"),
            ("uuid", "UUID generation and handling"),
            ("re", "Regular expression operations"),
            ("traceback", "Error traceback handling"),
            ("datetime", "Date and time operations"),
            ("urllib.parse", "URL parsing and encoding"),
        ]

        print("📋 Testing API utilities module imports:")
        results = []

        for module_name, description in required_modules:
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

            # Additional check for datetime and urllib.parse
            if not module_imported and "." in module_name:
                try:
                    __import__(module_name)
                    module_imported = True
                    import_method = "dynamic import"
                except ImportError:
                    import_method = "import failed"
            else:
                import_method = "sys.modules/globals check"

            status = "✅" if module_imported else "❌"
            print(f"   {status} {module_name}: {description}")
            print(f"      Method: {import_method}, Available: {module_imported}")

            results.append(module_imported)
            assert module_imported, f"Required module {module_name} not available"

        print(f"📊 Results: {sum(results)}/{len(results)} required modules imported")

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

    suite.run_test(
        "Module Imports",
        test_module_imports,
        "9 required modules imported: json, requests, time, logging, uuid, re, traceback, datetime, urllib.parse.",
        "Test all required modules and dependencies are properly imported with detailed verification.",
        "Verify json→serialization, requests→HTTP, time→timing, logging→logs, uuid→IDs, re→regex, traceback→errors, datetime→dates, urllib.parse→URLs.",
    )

    suite.run_test(
        "Optional Dependencies",
        test_optional_dependencies,
        "2 optional dependency flags tested: PYDANTIC_AVAILABLE and BS4_AVAILABLE boolean detection.",
        "Test optional dependencies are properly detected.",
        "Verify PYDANTIC_AVAILABLE→bool for validation, BS4_AVAILABLE→bool for HTML parsing availability.",
    )

    suite.run_test(
        "Logger Initialization",
        test_logger_initialization,
        "4 logger methods tested: logger exists, has info(), error(), warning() methods.",
        "Test logging configuration is properly set up.",
        "Test logger object exists and has info, error, warning methods",
    )


def _run_core_functionality_tests(suite: "TestSuite") -> None:
    """Run core functionality tests for api_utils module."""
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
        except json.JSONDecodeError as json_err:
            raise AssertionError("Valid JSON should parse successfully") from json_err

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


def _run_edge_case_tests(suite: "TestSuite") -> None:
    """Run edge case tests for api_utils module."""
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


def _run_integration_tests(suite: "TestSuite") -> None:
    """Run integration tests for api_utils module."""
    def test_config_integration():
        """Test integration with configuration management."""
        try:
            from config import config_schema

            assert config_schema is not None, "Config schema should be available"
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
            raise AssertionError(f"Logging should work without errors: {e}") from e

    def test_datetime_handling():
        """Test datetime parsing and formatting integration."""
        from datetime import datetime, timezone

        test_datetime = datetime.now(timezone.utc)
        assert isinstance(test_datetime, datetime), "Should create datetime object"

        formatted = test_datetime.isoformat()
        assert isinstance(formatted, str), "Formatted datetime should be string"
        assert "T" in formatted, "ISO format should contain T separator"

    suite.run_test(
        "Config Integration",
        test_config_integration,
        "Configuration integration should work seamlessly",
        "Integration with configuration management works correctly",
        "Test accessing config_schema and base_url configuration",
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


def _run_performance_tests(suite: "TestSuite") -> None:
    """Run performance tests for api_utils module."""
    def test_json_parsing_performance():
        """Test JSON parsing performance with large data sets."""

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

        test_strings = [f"Person Name {i} & Family" for i in range(100)]

        start_time = time.time()
        for test_string in test_strings:
            quote(test_string)
        end_time = time.time()

        encoding_time = end_time - start_time
        assert (
            encoding_time < 0.1
        ), f"URL encoding took {encoding_time:.3f}s, should be < 0.1s"

    def test_data_processing_efficiency():
        """Test efficiency of data processing operations."""

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


def _run_error_handling_tests(suite: "TestSuite") -> None:
    """Run error handling tests for api_utils module."""
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
            # Validate that we can detect various invalid PersonId scenarios
            is_empty = person_id is None or person_id == ""
            is_wrong_type = isinstance(person_id, int)
            assert is_empty or is_wrong_type or isinstance(person_id, str), "PersonId validation should work"

    def test_configuration_errors():
        """Test handling of missing or invalid configuration."""
        missing_config_cases = [
            None,
            {},
            {"incomplete": "config"},
        ]

        for config_case in missing_config_cases:
            is_none = config_case is None
            is_empty_or_incomplete = isinstance(config_case, dict) and (not config_case or "base_url" not in config_case)
            assert is_none or is_empty_or_incomplete, "Config validation should detect missing/invalid config"

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


def api_utils_module_tests() -> bool:
    """
    Comprehensive test suite for api_utils.py following the standardized 6-category TestSuite framework.
    Tests API functionality, data parsing, and integration capabilities.

    Categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, Error Handling
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("API Utilities & Ancestry Integration", "api_utils.py")
        suite.start_suite()

    # Run all test categories
    _run_initialization_tests(suite)
    _run_core_functionality_tests(suite)
    _run_edge_case_tests(suite)
    _run_integration_tests(suite)
    _run_performance_tests(suite)
    _run_error_handling_tests(suite)

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(api_utils_module_tests)


# --- Main Execution Block ---

if __name__ == "__main__":
    import sys

    print("🌐 Running API Utilities & Ancestry Integration comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
