#!/usr/bin/env python3
"""
Person Lookup Utilities for Phase 2: Person Lookup Integration

Provides data structures and functions for looking up people mentioned in messages
using Action 10 (GEDCOM) and Action 11 (API).
"""

# === STANDARD IMPORTS ===
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from standard_imports import setup_module

logger = setup_module(globals(), __file__)


@dataclass
class PersonLookupResult:
    """
    Result of looking up a person mentioned in a message.
    Contains genealogical details and relationship information.
    """

    # Basic identification
    person_id: str | None = None  # GEDCOM ID or API person ID
    name: str = ""  # Full name
    first_name: str | None = None
    last_name: str | None = None

    # Birth information
    birth_year: int | None = None
    birth_place: str | None = None
    birth_date: str | None = None  # Full date if available

    # Death information
    death_year: int | None = None
    death_place: str | None = None
    death_date: str | None = None  # Full date if available

    # Gender
    gender: str | None = None  # 'M' or 'F'

    # Relationship information
    relationship_path: str | None = None  # e.g., "3rd great-grandfather"
    relationship_description: str | None = None  # Detailed path

    # Family details
    family_details: dict[str, Any] = field(default_factory=dict)  # Parents, spouse, children

    # Match quality
    match_score: int = 0  # Scoring from Action 10/11
    confidence: str = "unknown"  # low, medium, high

    # Source
    source: str = "unknown"  # 'gedcom', 'api', or 'not_found'
    source_details: str | None = None  # Additional source information

    # Additional data
    notes: str | None = None
    found: bool = False  # Whether person was found

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "birth_year": self.birth_year,
            "birth_place": self.birth_place,
            "birth_date": self.birth_date,
            "death_year": self.death_year,
            "death_place": self.death_place,
            "death_date": self.death_date,
            "gender": self.gender,
            "relationship_path": self.relationship_path,
            "relationship_description": self.relationship_description,
            "family_details": self.family_details,
            "match_score": self.match_score,
            "confidence": self.confidence,
            "source": self.source,
            "source_details": self.source_details,
            "notes": self.notes,
            "found": self.found,
        }

    def _format_birth_info(self) -> str | None:
        """Format birth information."""
        if self.birth_year and self.birth_place:
            return f"born {self.birth_year} in {self.birth_place}"
        if self.birth_year:
            return f"born {self.birth_year}"
        if self.birth_place:
            return f"born in {self.birth_place}"
        return None

    def _format_death_info(self) -> str | None:
        """Format death information."""
        if self.death_year and self.death_place:
            return f"died {self.death_year} in {self.death_place}"
        if self.death_year:
            return f"died {self.death_year}"
        return None

    def _format_family_details(self) -> list[str]:
        """Format family details."""
        parts = []
        if not self.family_details:
            return parts

        if "parents" in self.family_details:
            parents = self.family_details["parents"]
            if parents:
                parts.append(f"parents: {', '.join(parents)}")

        if "spouse" in self.family_details:
            spouse = self.family_details["spouse"]
            if spouse:
                parts.append(f"spouse: {spouse}")

        if "children" in self.family_details:
            children = self.family_details["children"]
            if children:
                parts.append(f"children: {', '.join(children[:3])}")

        return parts

    def format_for_ai(self) -> str:
        """Format lookup result for inclusion in AI prompt."""
        if not self.found:
            return f"Person '{self.name}' not found in tree or records."

        parts = [f"Found: {self.name}"]

        # Add birth info
        birth_info = self._format_birth_info()
        if birth_info:
            parts.append(birth_info)

        # Add death info
        death_info = self._format_death_info()
        if death_info:
            parts.append(death_info)

        # Add relationship
        if self.relationship_path:
            parts.append(f"relationship: {self.relationship_path}")

        # Add family details
        parts.extend(self._format_family_details())

        return ", ".join(parts)

    def __str__(self) -> str:
        """String representation for logging."""
        if not self.found:
            return f"PersonLookupResult(name='{self.name}', found=False)"

        return (
            f"PersonLookupResult(name='{self.name}', "
            f"birth={self.birth_year or 'unknown'}, "
            f"relationship='{self.relationship_path or 'unknown'}', "
            f"source='{self.source}', "
            f"score={self.match_score})"
        )


def create_not_found_result(name: str, reason: str | None = None) -> PersonLookupResult:
    """Create a PersonLookupResult for a person not found."""
    return PersonLookupResult(
        name=name,
        found=False,
        source="not_found",
        notes=reason or "Person not found in tree or records",
        confidence="low",
    )


def create_result_from_gedcom(
    person_data: dict[str, Any],
    relationship_path: str | None = None,
    match_score: int = 0,
) -> PersonLookupResult:
    """Create PersonLookupResult from GEDCOM search result (Action 10)."""
    return PersonLookupResult(
        person_id=person_data.get("id"),
        name=person_data.get("name", ""),
        first_name=person_data.get("first_name"),
        last_name=person_data.get("surname"),
        birth_year=person_data.get("birth_year"),
        birth_place=person_data.get("birth_place"),
        birth_date=person_data.get("birth_date"),
        death_year=person_data.get("death_year"),
        death_place=person_data.get("death_place"),
        death_date=person_data.get("death_date"),
        gender=person_data.get("gender"),
        relationship_path=relationship_path,
        family_details=person_data.get("family", {}),
        match_score=match_score,
        confidence="high" if match_score > 200 else "medium" if match_score > 100 else "low",
        source="gedcom",
        source_details="Found in GEDCOM file",
        found=True,
    )


def create_result_from_api(
    person_data: dict[str, Any],
    relationship_path: str | None = None,
    match_score: int = 0,
) -> PersonLookupResult:
    """Create PersonLookupResult from API search result (Action 11)."""
    return PersonLookupResult(
        person_id=person_data.get("personId"),
        name=person_data.get("name", ""),
        first_name=person_data.get("givenName"),
        last_name=person_data.get("surname"),
        birth_year=person_data.get("birthYear"),
        birth_place=person_data.get("birthPlace"),
        death_year=person_data.get("deathYear"),
        death_place=person_data.get("deathPlace"),
        gender=person_data.get("gender"),
        relationship_path=relationship_path,
        family_details=person_data.get("family", {}),
        match_score=match_score,
        confidence="high" if match_score > 200 else "medium" if match_score > 100 else "low",
        source="api",
        source_details="Found via Ancestry API",
        found=True,
    )


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def person_lookup_utils_module_tests() -> bool:
    """
    Comprehensive test suite for person_lookup_utils.py.
    Tests PersonLookupResult data structure and factory functions.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Person Lookup Utils Tests", "person_lookup_utils.py")
    suite.start_suite()

    with suppress_logging():
        # Test 1: PersonLookupResult creation and basic properties
        suite.run_test(
            "PersonLookupResult creation",
            _test_person_lookup_result_creation,
            test_summary="Create PersonLookupResult with various data combinations",
            functions_tested="PersonLookupResult.__init__(), to_dict(), __str__()",
            method_description="Create result objects with different field combinations and verify properties",
            expected_outcome="All fields correctly initialized, to_dict() returns complete dictionary",
        )

        # Test 2: format_for_ai() method
        suite.run_test(
            "AI formatting",
            _test_format_for_ai,
            test_summary="Test format_for_ai() with found and not-found results",
            functions_tested="PersonLookupResult.format_for_ai()",
            method_description="Format person data for AI consumption with various field combinations",
            expected_outcome="Properly formatted strings for AI prompts with all available data",
        )

        # Test 3: create_not_found_result()
        suite.run_test(
            "Not found result creation",
            _test_create_not_found_result,
            test_summary="Create not-found results with various reasons",
            functions_tested="create_not_found_result()",
            method_description="Create PersonLookupResult for people not found in tree/API",
            expected_outcome="Result with found=False, appropriate source and notes",
        )

        # Test 4: create_result_from_gedcom()
        suite.run_test(
            "GEDCOM result creation",
            _test_create_result_from_gedcom,
            test_summary="Create results from GEDCOM search data",
            functions_tested="create_result_from_gedcom()",
            method_description="Convert Action 10 GEDCOM search results to PersonLookupResult",
            expected_outcome="Result with source='gedcom', correct confidence based on score",
        )

        # Test 5: create_result_from_api()
        suite.run_test(
            "API result creation",
            _test_create_result_from_api,
            test_summary="Create results from API search data",
            functions_tested="create_result_from_api()",
            method_description="Convert Action 11 API search results to PersonLookupResult",
            expected_outcome="Result with source='api', correct field mapping from API response",
        )

        # Test 6: Confidence scoring
        suite.run_test(
            "Confidence scoring",
            _test_confidence_scoring,
            test_summary="Verify confidence levels based on match scores",
            functions_tested="create_result_from_gedcom(), create_result_from_api()",
            method_description="Test confidence assignment: high (>200), medium (>100), low (<=100)",
            expected_outcome="Correct confidence levels for different score ranges",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return person_lookup_utils_module_tests()


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def _test_person_lookup_result_creation() -> bool:
    """Test PersonLookupResult creation and basic properties."""
    # Test 1: Minimal creation
    result1 = PersonLookupResult(name="John Smith")
    assert result1.name == "John Smith"
    assert result1.found is False
    assert result1.match_score == 0

    # Test 2: Full creation
    result2 = PersonLookupResult(
        person_id="I123",
        name="Jane Doe",
        first_name="Jane",
        last_name="Doe",
        birth_year=1850,
        birth_place="London, England",
        death_year=1920,
        death_place="New York, USA",
        gender="F",
        relationship_path="3rd great-grandmother",
        match_score=150,
        confidence="medium",
        source="gedcom",
        found=True,
    )
    assert result2.person_id == "I123"
    assert result2.birth_year == 1850
    assert result2.found is True

    # Test 3: to_dict() conversion
    result_dict = result2.to_dict()
    assert result_dict["name"] == "Jane Doe"
    assert result_dict["birth_year"] == 1850
    assert result_dict["found"] is True

    # Test 4: __str__() representation
    str_repr = str(result2)
    assert "Jane Doe" in str_repr
    assert "1850" in str_repr

    return True


def _test_format_for_ai() -> bool:
    """Test format_for_ai() method."""
    # Test 1: Not found result
    not_found = PersonLookupResult(name="Unknown Person", found=False)
    formatted = not_found.format_for_ai()
    assert "not found" in formatted.lower()
    assert "Unknown Person" in formatted

    # Test 2: Found with birth/death info
    found_person = PersonLookupResult(
        name="Charles Fetch",
        birth_year=1881,
        birth_place="Banff, Scotland",
        death_year=1948,
        relationship_path="great-grandfather",
        found=True,
    )
    formatted = found_person.format_for_ai()
    assert "Charles Fetch" in formatted
    assert "1881" in formatted
    assert "Banff, Scotland" in formatted
    assert "1948" in formatted
    assert "great-grandfather" in formatted

    # Test 3: With family details
    with_family = PersonLookupResult(
        name="Mary MacDonald",
        family_details={
            "parents": ["William MacDonald", "Margaret Smith"],
            "spouse": "Charles Fetch",
            "children": ["Helen Fetch", "James Fetch", "Margaret Fetch"],
        },
        found=True,
    )
    formatted = with_family.format_for_ai()
    assert "William MacDonald" in formatted
    assert "Charles Fetch" in formatted
    assert "Helen Fetch" in formatted

    return True


def _test_create_not_found_result() -> bool:
    """Test create_not_found_result() function."""
    # Test 1: Basic not found
    result1 = create_not_found_result("John Unknown")
    assert result1.name == "John Unknown"
    assert result1.found is False
    assert result1.source == "not_found"
    assert result1.confidence == "low"

    # Test 2: With custom reason
    result2 = create_not_found_result("Jane Unknown", reason="No matching records in GEDCOM")
    assert result2.notes == "No matching records in GEDCOM"

    return True


def _test_create_result_from_gedcom() -> bool:
    """Test create_result_from_gedcom() function."""
    gedcom_data = {
        "id": "I456",
        "name": "James Gault",
        "first_name": "James",
        "surname": "Gault",
        "birth_year": 1885,
        "birth_place": "Banff, Scotland",
        "death_year": 1962,
        "gender": "M",
        "family": {"spouse": "Margaret Milne", "children": ["Helen Gault"]},
    }

    result = create_result_from_gedcom(
        gedcom_data,
        relationship_path="2nd great-grandfather",
        match_score=180,
    )

    assert result.person_id == "I456"
    assert result.name == "James Gault"
    assert result.first_name == "James"
    assert result.last_name == "Gault"
    assert result.birth_year == 1885
    assert result.source == "gedcom"
    assert result.found is True
    assert result.match_score == 180
    assert result.confidence == "medium"  # 100 < 180 <= 200
    assert result.relationship_path == "2nd great-grandfather"

    return True


def _test_create_result_from_api() -> bool:
    """Test create_result_from_api() function."""
    api_data = {
        "personId": "P789",
        "name": "William MacDonald",
        "givenName": "William",
        "surname": "MacDonald",
        "birthYear": 1850,
        "birthPlace": "Aberdeen, Scotland",
        "deathYear": 1920,
        "gender": "M",
        "family": {"spouse": "Margaret Smith"},
    }

    result = create_result_from_api(
        api_data,
        relationship_path="3rd great-grandfather",
        match_score=220,
    )

    assert result.person_id == "P789"
    assert result.name == "William MacDonald"
    assert result.first_name == "William"
    assert result.last_name == "MacDonald"
    assert result.birth_year == 1850
    assert result.source == "api"
    assert result.found is True
    assert result.match_score == 220
    assert result.confidence == "high"  # 220 > 200
    assert result.source_details == "Found via Ancestry API"

    return True


def _test_confidence_scoring() -> bool:
    """Test confidence level assignment based on match scores."""
    # Test high confidence (>200)
    high_gedcom = create_result_from_gedcom({"name": "Test"}, match_score=250)
    assert high_gedcom.confidence == "high"

    high_api = create_result_from_api({"name": "Test"}, match_score=300)
    assert high_api.confidence == "high"

    # Test medium confidence (100 < score <= 200)
    medium_gedcom = create_result_from_gedcom({"name": "Test"}, match_score=150)
    assert medium_gedcom.confidence == "medium"

    medium_api = create_result_from_api({"name": "Test"}, match_score=101)
    assert medium_api.confidence == "medium"

    # Test low confidence (<=100)
    low_gedcom = create_result_from_gedcom({"name": "Test"}, match_score=50)
    assert low_gedcom.confidence == "low"

    low_api = create_result_from_api({"name": "Test"}, match_score=100)
    assert low_api.confidence == "low"

    return True


if __name__ == "__main__":
    import sys
    import traceback

    try:
        print("🧪 Running Person Lookup Utils comprehensive test suite...")
        success = run_comprehensive_tests()
    except Exception:
        print("\n[ERROR] Unhandled exception during Person Lookup Utils tests:", file=sys.stderr)
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
