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
MatchScore = int | float
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
# Main protocols (RateLimiterProtocol, CacheProtocol, SessionManagerProtocol)
# are defined in core/protocols.py - import them from there.
# These are domain-specific protocols that use the types from this module.


class Loggable(Protocol):
    """Protocol for objects that can be logged."""

    def __str__(self) -> str: ...


class Cacheable(Protocol):
    """Protocol for objects that can be cached."""

    def cache_key(self) -> CacheKey: ...


class Scoreable(Protocol):
    """Protocol for objects that can be scored."""

    def calculate_score(self, criteria: SearchCriteria) -> MatchScore: ...


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

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    from testing.test_framework import TestSuite

    suite = TestSuite("Type Definitions", "core/type_definitions.py")
    suite.start_suite()

    def test_type_aliases_exist():
        assert PersonId is str
        assert TreeId is str
        assert ProfileId is str
        assert UUID is str
        assert CacheKey is str
        assert CacheTTL is int
        assert StatusCode is int
        assert QualityScore is int
        return True

    suite.run_test("Type aliases are valid", test_type_aliases_exist)

    def test_person_info_instantiation():
        p: PersonInfo = {"person_id": "I123", "full_name": "John Doe", "birth_year": 1900}
        assert p["person_id"] == "I123"
        assert p["full_name"] == "John Doe"
        assert p["birth_year"] == 1900
        return True

    suite.run_test("PersonInfo TypedDict instantiation", test_person_info_instantiation)

    def test_dna_match_info_instantiation():
        d: DNAMatchInfo = {"uuid": "ABC-123", "name": "Jane", "shared_cm": 250.5, "shared_segments": 12}
        assert d["uuid"] == "ABC-123"
        assert d["shared_cm"] == 250.5
        return True

    suite.run_test("DNAMatchInfo TypedDict instantiation", test_dna_match_info_instantiation)

    def test_api_response_instantiation():
        r: APIResponse = {"success": True, "status_code": 200, "data": {"key": "val"}, "error": None}
        assert r["success"] is True
        assert r["status_code"] == 200
        return True

    suite.run_test("APIResponse TypedDict instantiation", test_api_response_instantiation)

    def test_session_state_instantiation():
        s: SessionState = {"is_valid": True, "browser_ready": False, "session_age_seconds": 42.0}
        assert s["is_valid"] is True
        assert s["session_age_seconds"] == 42.0
        return True

    suite.run_test("SessionState TypedDict instantiation", test_session_state_instantiation)

    def test_search_criteria_and_result():
        sc: SearchCriteria = {"first_name": "John", "surname": "Doe", "birth_year": 1900}
        assert sc["first_name"] == "John"
        sr: SearchResult = {"person_id": "I1", "tree_id": "T1", "name": "John Doe", "score": 95}
        assert sr["score"] == 95
        return True

    suite.run_test("SearchCriteria and SearchResult instantiation", test_search_criteria_and_result)

    def test_constants():
        assert STATUS_SUCCESS == "success"
        assert STATUS_ERROR == "error"
        assert STATUS_PENDING == "pending"
        assert GENDER_MALE == "m"
        assert GENDER_FEMALE == "f"
        assert "vital_records" in TASK_CATEGORIES
        assert "dna_analysis" in TASK_CATEGORIES
        assert len(TASK_CATEGORIES) == 8
        return True

    suite.run_test("Constants are defined correctly", test_constants)

    def test_http_method_literal():
        valid_methods: list[HTTPMethod] = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        assert len(valid_methods) == 5
        return True

    suite.run_test("HTTPMethod literal values", test_http_method_literal)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
