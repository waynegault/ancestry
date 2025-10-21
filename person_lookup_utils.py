#!/usr/bin/env python3
"""
Person Lookup Utilities for Phase 2: Person Lookup Integration

Provides data structures and functions for looking up people mentioned in messages
using Action 10 (GEDCOM) and Action 11 (API).
"""

# === STANDARD IMPORTS ===
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
    person_id: Optional[str] = None  # GEDCOM ID or API person ID
    name: str = ""  # Full name
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    # Birth information
    birth_year: Optional[int] = None
    birth_place: Optional[str] = None
    birth_date: Optional[str] = None  # Full date if available

    # Death information
    death_year: Optional[int] = None
    death_place: Optional[str] = None
    death_date: Optional[str] = None  # Full date if available

    # Gender
    gender: Optional[str] = None  # 'M' or 'F'

    # Relationship information
    relationship_path: Optional[str] = None  # e.g., "3rd great-grandfather"
    relationship_description: Optional[str] = None  # Detailed path

    # Family details
    family_details: dict[str, Any] = field(default_factory=dict)  # Parents, spouse, children

    # Match quality
    match_score: int = 0  # Scoring from Action 10/11
    confidence: str = "unknown"  # low, medium, high

    # Source
    source: str = "unknown"  # 'gedcom', 'api', or 'not_found'
    source_details: Optional[str] = None  # Additional source information

    # Additional data
    notes: Optional[str] = None
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

    def format_for_ai(self) -> str:
        """Format lookup result for inclusion in AI prompt."""
        if not self.found:
            return f"Person '{self.name}' not found in tree or records."

        parts = [f"Found: {self.name}"]

        # Add birth info
        if self.birth_year and self.birth_place:
            parts.append(f"born {self.birth_year} in {self.birth_place}")
        elif self.birth_year:
            parts.append(f"born {self.birth_year}")
        elif self.birth_place:
            parts.append(f"born in {self.birth_place}")

        # Add death info
        if self.death_year and self.death_place:
            parts.append(f"died {self.death_year} in {self.death_place}")
        elif self.death_year:
            parts.append(f"died {self.death_year}")

        # Add relationship
        if self.relationship_path:
            parts.append(f"relationship: {self.relationship_path}")

        # Add family details
        if self.family_details:
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
                    parts.append(f"children: {', '.join(children[:3])}")  # Limit to 3

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


def create_not_found_result(name: str, reason: Optional[str] = None) -> PersonLookupResult:
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
    relationship_path: Optional[str] = None,
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
    relationship_path: Optional[str] = None,
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

