"""Shared type definitions for the Ancestry Research Automation Platform.

This module contains common type aliases, TypedDicts, and Protocol definitions
that are shared across multiple modules. Placing these here helps avoid
circular imports by providing a single source of truth for shared types.

Usage:
    from core.type_definitions import (
        PersonId,
        TreeId,
        SessionState,
        APIResponse,
        CacheKey,
    )
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, Union

# =============================================================================
# Simple Type Aliases
# =============================================================================

# Identity types - use NewType in Python 3.10+ for stronger typing
PersonId = str  # Ancestry person identifier (e.g., "I102281560744")
TreeId = str  # Ancestry tree identifier
ProfileId = str  # Ancestry profile/user identifier
UUID = str  # DNA test GUID (uppercase)
ConversationId = str  # Messaging conversation identifier

# Score types
MatchScore = Union[int, float]
QualityScore = int  # 0-100 scale

# Date types
DateString = str  # Formatted date like "15 June 1941"
NormalizedDate = str  # ISO format like "1941-06-15"
BirthYear = int
DeathYear = int

# Place types
PlaceString = str  # Place name like "Banff, Banffshire, Scotland"

# API types
HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
StatusCode = int

# Cache types
CacheKey = str
CacheTTL = int  # Seconds


# =============================================================================
# TypedDicts for Structured Data
# =============================================================================


class PersonInfo(TypedDict, total=False):
    """Basic person information structure."""

    person_id: PersonId
    tree_id: TreeId
    full_name: str
    given_name: str
    surname: str
    birth_year: int | None
    birth_date: DateString | None
    birth_place: PlaceString | None
    death_year: int | None
    death_date: DateString | None
    death_place: PlaceString | None
    gender: str
    is_living: bool


class DNAMatchInfo(TypedDict, total=False):
    """DNA match information structure."""

    uuid: UUID
    name: str
    shared_cm: float
    shared_segments: int
    relationship_range: str
    profile_id: ProfileId | None
    administrator_profile_id: ProfileId | None
    has_tree: bool
    tree_size: int | None


class EventInfo(TypedDict, total=False):
    """Event information (birth, death, etc.)."""

    year: int | None
    date: DateString | None
    place: PlaceString | None


class SearchCriteria(TypedDict, total=False):
    """Search criteria for finding people."""

    first_name: str
    surname: str
    birth_year: int | None
    birth_place: str | None
    death_year: int | None
    death_place: str | None
    gender: str | None


class SearchResult(TypedDict, total=False):
    """Search result with score."""

    person_id: PersonId
    tree_id: TreeId
    name: str
    birth_date: DateString | None
    birth_place: PlaceString | None
    death_date: DateString | None
    death_place: PlaceString | None
    score: MatchScore
    match_reasons: list[str]


class APIResponse(TypedDict, total=False):
    """Generic API response structure."""

    success: bool
    status_code: StatusCode
    data: Any
    error: str | None
    retry_after: int | None


class SessionState(TypedDict, total=False):
    """Session state information."""

    is_valid: bool
    browser_ready: bool
    api_ready: bool
    db_ready: bool
    session_age_seconds: float
    last_activity: str


class ConversationMessage(TypedDict, total=False):
    """Conversation message structure."""

    message_id: str
    conversation_id: ConversationId
    author_profile_id: ProfileId
    content: str
    timestamp: str
    is_outbound: bool


class TaskInfo(TypedDict, total=False):
    """Task information for MS To-Do integration."""

    title: str
    body: str
    category: str
    priority: Literal["low", "normal", "high"]
    due_date: str | None
    source_conversation_id: ConversationId | None


# =============================================================================
# Protocols for Duck Typing
# =============================================================================


class Loggable(Protocol):
    """Protocol for objects that can be logged."""

    def __str__(self) -> str: ...


class Cacheable(Protocol):
    """Protocol for objects that can be cached."""

    def cache_key(self) -> CacheKey: ...


class Scoreable(Protocol):
    """Protocol for objects that can be scored."""

    def calculate_score(self, criteria: SearchCriteria) -> MatchScore: ...


class SessionManagerProtocol(Protocol):
    """Protocol for session manager interface.

    This allows type checking without importing the actual SessionManager,
    which helps avoid circular imports.
    """

    def is_sess_valid(self) -> bool: ...
    def ensure_session_ready(self) -> bool: ...
    def ensure_db_ready(self) -> bool: ...
    def session_age_seconds(self) -> float: ...


class RateLimiterProtocol(Protocol):
    """Protocol for rate limiter interface."""

    def wait(self) -> None: ...
    def reset(self) -> None: ...


class CacheProtocol(Protocol):
    """Protocol for cache interface."""

    def get(self, key: CacheKey) -> Any | None: ...
    def set(self, key: CacheKey, value: Any, ttl: CacheTTL | None = None) -> None: ...
    def delete(self, key: CacheKey) -> bool: ...
    def clear(self) -> None: ...


# =============================================================================
# Constants
# =============================================================================

# Status constants
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_PENDING = "pending"
STATUS_SKIPPED = "skipped"

# Gender constants
GENDER_MALE = "m"
GENDER_FEMALE = "f"
GENDER_UNKNOWN = "u"

# Message classification labels
CLASSIFICATION_PRODUCTIVE = "PRODUCTIVE"
CLASSIFICATION_ENTHUSIASTIC = "ENTHUSIASTIC"
CLASSIFICATION_DESIST = "DESIST"
CLASSIFICATION_OTHER = "OTHER"

# Task categories
TASK_CATEGORIES = frozenset(
    {
        "vital_records",
        "census_research",
        "dna_analysis",
        "document_collection",
        "family_interview",
        "location_research",
        "relationship_verification",
        "general_research",
    }
)
