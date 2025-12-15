#!/usr/bin/env python3
"""
Fact Validation Pipeline

Validates genealogical facts extracted from conversations, detects conflicts with
existing data, and manages the approval workflow per data_validation_pipeline.md spec.

Phase 2 Implementation (Dec 2025):
- ExtractedFact dataclass for runtime fact representation
- ExistingFact dataclass for tree/database facts
- FactValidator service with conflict detection
- ConflictType enum for categorizing discrepancies
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy.orm import Session as DbSession

if TYPE_CHECKING:
    from core.database import Person

# === MODULE SETUP ===
logger = logging.getLogger(__name__)


# === ENUMS ===


class ConflictType(Enum):
    """Types of conflicts between extracted and existing facts."""

    EXACT_MATCH = "EXACT_MATCH"  # New fact matches existing exactly
    COMPATIBLE = "COMPATIBLE"  # New fact adds detail to existing
    MINOR_CONFLICT = "MINOR_CONFLICT"  # Small discrepancy (1-2 years, spelling variant)
    MAJOR_CONFLICT = "MAJOR_CONFLICT"  # Significant disagreement (>5 years, different name)
    NO_EXISTING = "NO_EXISTING"  # No prior value exists


class ReviewPriority(Enum):
    """Priority levels for human review queue."""

    LOW = "LOW"  # Auto-approved, logged for reference
    NORMAL = "NORMAL"  # Standard review
    HIGH = "HIGH"  # Conflict detected, needs attention
    CRITICAL = "CRITICAL"  # Major conflict or low confidence


# === DATACLASSES ===


@dataclass()
class ExtractedFact:
    """
    Runtime representation of a fact extracted from conversation.

    This is the standardized output format for Fact Extraction 2.0.
    All facts extracted by AI should be converted to this format.
    """

    fact_type: str  # BIRTH, DEATH, RELATIONSHIP, MARRIAGE, LOCATION, OTHER
    subject_name: str  # Who is this fact about? (full name)
    original_text: str  # "My grandmother Mary was born in 1920"
    structured_value: str  # "1920" or "Mary Ellen Smith"
    normalized_value: str  # "1920-01-01" (ISO date) or standardized name
    confidence: int  # 0-100
    source_conversation_id: Optional[str] = None  # Traceability
    extraction_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional enrichment fields
    subject_person_id: Optional[int] = None  # Linked person in DB (if found)
    related_person_name: Optional[str] = None  # For RELATIONSHIP facts
    location: Optional[str] = None  # Place associated with fact
    date_qualifier: Optional[str] = None  # "circa", "before", "after", "about"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fact_type": self.fact_type,
            "subject_name": self.subject_name,
            "original_text": self.original_text,
            "structured_value": self.structured_value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_conversation_id": self.source_conversation_id,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "subject_person_id": self.subject_person_id,
            "related_person_name": self.related_person_name,
            "location": self.location,
            "date_qualifier": self.date_qualifier,
        }

    @classmethod
    def from_vital_record(
        cls,
        person_name: str,
        event_type: str,
        date: str,
        place: str,
        certainty: str,
        original_text: str,
        conversation_id: Optional[str] = None,
    ) -> ExtractedFact:
        """Factory method to create from VitalRecord extraction."""
        # Map event_type to fact_type
        type_map = {
            "birth": "BIRTH",
            "death": "DEATH",
            "marriage": "MARRIAGE",
            "baptism": "OTHER",
            "burial": "OTHER",
        }
        fact_type = type_map.get(event_type.lower(), "OTHER")

        # Map certainty to confidence score
        certainty_map = {"certain": 95, "probable": 75, "uncertain": 50}
        confidence = certainty_map.get(certainty.lower(), 70)

        # Normalize the date and extract qualifier
        normalized = cls._normalize_date(date)
        qualifier = cls._extract_date_qualifier(date)

        return cls(
            fact_type=fact_type,
            subject_name=person_name,
            original_text=original_text,
            structured_value=date,
            normalized_value=normalized,
            confidence=confidence,
            source_conversation_id=conversation_id,
            location=place if place else None,
            date_qualifier=qualifier,
        )

    @classmethod
    def from_relationship(
        cls,
        person1: str,
        relationship: str,
        person2: str,
        original_text: str,
        conversation_id: Optional[str] = None,
    ) -> ExtractedFact:
        """Factory method to create from Relationship extraction."""
        return cls(
            fact_type="RELATIONSHIP",
            subject_name=person1,
            original_text=original_text,
            structured_value=relationship,
            normalized_value=relationship.lower(),
            confidence=85,  # Relationships are generally reliable when stated
            source_conversation_id=conversation_id,
            related_person_name=person2,
        )

    @classmethod
    def from_conversation(
        cls,
        conversation_id: str,
        extracted_data: dict[str, Any],
        original_text: str = "",
    ) -> list[ExtractedFact]:
        """
        Factory method to create ExtractedFact objects from AI extraction output.

        This is the main entry point for converting AI-extracted data
        (from the extraction_task prompt) into standardized ExtractedFact objects.

        Args:
            conversation_id: ID of the source conversation
            extracted_data: Dictionary from AI extraction (matches extraction_task schema)
            original_text: Original message text for context

        Returns:
            List of ExtractedFact objects extracted from the data
        """
        facts: list[ExtractedFact] = []

        # Extract from vital_records
        for record in extracted_data.get("vital_records", []):
            fact = cls.from_vital_record(
                person_name=record.get("person", "Unknown"),
                event_type=record.get("event_type", "OTHER"),
                date=record.get("date", ""),
                place=record.get("place", ""),
                certainty=record.get("certainty", "probable"),
                original_text=original_text,
                conversation_id=conversation_id,
            )
            facts.append(fact)

        # Extract from relationships
        for rel in extracted_data.get("relationships", []):
            fact = cls.from_relationship(
                person1=rel.get("person1", "Unknown"),
                relationship=rel.get("relationship", "related to"),
                person2=rel.get("person2", "Unknown"),
                original_text=original_text,
                conversation_id=conversation_id,
            )
            facts.append(fact)

        # Extract from mentioned_people (as person facts)
        for person in extracted_data.get("mentioned_people", []):
            name = person.get("name", "")
            if not name:
                continue

            # Create BIRTH fact if birth year available
            birth_year = person.get("birth_year")
            if birth_year:
                facts.append(
                    cls(
                        fact_type="BIRTH",
                        subject_name=name,
                        original_text=original_text,
                        structured_value=str(birth_year),
                        normalized_value=f"{birth_year}-01-01",
                        confidence=80,
                        source_conversation_id=conversation_id,
                        location=person.get("birth_place"),
                    )
                )

            # Create DEATH fact if death year available
            death_year = person.get("death_year")
            if death_year:
                facts.append(
                    cls(
                        fact_type="DEATH",
                        subject_name=name,
                        original_text=original_text,
                        structured_value=str(death_year),
                        normalized_value=f"{death_year}-01-01",
                        confidence=80,
                        source_conversation_id=conversation_id,
                        location=person.get("death_place"),
                    )
                )

            # Create LOCATION fact if location available (residence context)
            for loc in extracted_data.get("locations", []):
                if loc.get("context") == "residence" and loc.get("place"):
                    facts.append(
                        cls(
                            fact_type="LOCATION",
                            subject_name=name,
                            original_text=original_text,
                            structured_value=loc.get("place", ""),
                            normalized_value=loc.get("place", ""),
                            confidence=70,
                            source_conversation_id=conversation_id,
                            location=loc.get("place"),
                        )
                    )

        return facts

    @staticmethod
    def _extract_date_qualifier(date_str: str) -> Optional[str]:
        """Extract date qualifier (circa, before, after) from date string."""
        if not date_str:
            return None

        date_str = date_str.strip().lower()

        if re.match(r"^(?:circa|about|~|c\.?)\s*\d", date_str):
            return "circa"
        if re.match(r"^(?:before|bef\.?)\s*\d", date_str):
            return "before"
        if re.match(r"^(?:after|aft\.?)\s*\d", date_str):
            return "after"

        return None

    @staticmethod
    def _normalize_date(date_str: str) -> str:  # noqa: PLR0911 - multiple return paths for clarity
        """Normalize date string to ISO format where possible."""
        if not date_str:
            return ""

        date_str = date_str.strip()

        # Already ISO format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str

        # Year only (e.g., "1920")
        if re.match(r"^\d{4}$", date_str):
            return f"{date_str}-01-01"

        # Handle "circa", "about", "~" prefixes
        circa_match = re.match(r"^(?:circa|about|~|c\.?)\s*(\d{4})$", date_str, re.IGNORECASE)
        if circa_match:
            return f"{circa_match.group(1)}-01-01"

        # Handle "before", "after" prefixes
        before_match = re.match(r"^(?:before|bef\.?)\s*(\d{4})$", date_str, re.IGNORECASE)
        if before_match:
            return f"{before_match.group(1)}-01-01"

        after_match = re.match(r"^(?:after|aft\.?)\s*(\d{4})$", date_str, re.IGNORECASE)
        if after_match:
            return f"{after_match.group(1)}-12-31"

        # Return original if can't normalize
        return date_str


@dataclass()
class ExistingFact:
    """Representation of a fact already in the tree/database."""

    fact_type: str
    person_id: int
    person_name: str
    value: str
    source: str  # "GEDCOM", "User Input", "Conversation"
    last_updated: datetime


@dataclass()
class ValidationResult:
    """Result of validating an extracted fact."""

    fact: ExtractedFact
    conflict_type: ConflictType
    conflicting_fact: Optional[ExistingFact] = None
    auto_approved: bool = False
    review_priority: ReviewPriority = ReviewPriority.NORMAL
    suggested_fact_id: Optional[int] = None  # DB record ID if created
    message: str = ""


# === FACT VALIDATOR SERVICE ===


class FactValidator:
    """
    Service for validating extracted facts and detecting conflicts.

    Implements the validation pipeline from data_validation_pipeline.md:
    1. Check if existing fact exists
    2. Compare values and detect conflict type
    3. Determine if auto-approval is allowed
    4. Create SuggestedFact record with appropriate status
    """

    # Thresholds for auto-approval
    AUTO_APPROVE_MIN_CONFIDENCE = 80
    MINOR_CONFLICT_YEAR_DIFF = 2
    MAJOR_CONFLICT_YEAR_DIFF = 5
    NAME_SIMILARITY_THRESHOLD = 0.85

    def __init__(self, db_session: Optional[DbSession] = None):
        """Initialize with optional database session."""
        self.db_session = db_session

    def validate_fact(self, fact: ExtractedFact, person: Optional[Person] = None) -> ValidationResult:
        """
        Validate an extracted fact against existing data.

        Args:
            fact: The extracted fact to validate
            person: Optional Person object if already resolved

        Returns:
            ValidationResult with conflict type and approval status
        """
        # If no existing data to compare against, check confidence for auto-approve
        if person is None:
            if fact.confidence >= self.AUTO_APPROVE_MIN_CONFIDENCE:
                return ValidationResult(
                    fact=fact,
                    conflict_type=ConflictType.NO_EXISTING,
                    auto_approved=True,
                    review_priority=ReviewPriority.LOW,
                    message="No existing data, high confidence - auto-approved",
                )
            return ValidationResult(
                fact=fact,
                conflict_type=ConflictType.NO_EXISTING,
                auto_approved=False,
                review_priority=ReviewPriority.NORMAL,
                message="No existing data, moderate confidence - queued for review",
            )

        # Get existing facts for comparison (placeholder - would query DB in full impl)
        existing_facts = self._get_existing_facts(person, fact.fact_type)

        if not existing_facts:
            return self._handle_no_existing(fact)

        # Compare against each existing fact
        for existing in existing_facts:
            conflict = self._detect_conflict(fact, existing)
            if conflict != ConflictType.EXACT_MATCH:
                return self._create_result(fact, conflict, existing)

        # All matches are exact - update source only
        return ValidationResult(
            fact=fact,
            conflict_type=ConflictType.EXACT_MATCH,
            conflicting_fact=existing_facts[0] if existing_facts else None,
            auto_approved=True,
            review_priority=ReviewPriority.LOW,
            message="Exact match with existing fact - source updated",
        )

    def _get_existing_facts(self, person: Person, fact_type: str) -> list[ExistingFact]:
        """
        Get existing facts of the same type for a person from DB and SuggestedFact table.

        Sprint 3: Data Validation Pipeline - queries both person attributes and
        SuggestedFact table for comprehensive conflict detection.

        Args:
            person: Person object to check facts for
            fact_type: Type of fact (BIRTH, DEATH, etc.)

        Returns:
            List of ExistingFact objects for comparison
        """
        existing: list[ExistingFact] = []

        # 1. Check person's direct attributes based on fact type
        if fact_type == "BIRTH" and hasattr(person, "birth_year") and person.birth_year:
            existing.append(
                ExistingFact(
                    fact_type="BIRTH",
                    person_id=person.id,
                    person_name=person.display_name or str(person.id),
                    value=str(person.birth_year),
                    source="Database",
                    last_updated=datetime.now(timezone.utc),
                )
            )
        elif fact_type == "DEATH" and hasattr(person, "death_year") and person.death_year:
            existing.append(
                ExistingFact(
                    fact_type="DEATH",
                    person_id=person.id,
                    person_name=person.display_name or str(person.id),
                    value=str(person.death_year),
                    source="Database",
                    last_updated=datetime.now(timezone.utc),
                )
            )

        # 2. Query SuggestedFact table for approved facts
        if self.db_session:
            existing.extend(self._get_suggested_facts_from_db(person.id, fact_type))

        return existing

    def _get_suggested_facts_from_db(self, person_id: int, fact_type: str) -> list[ExistingFact]:
        """
        Query SuggestedFact table for approved facts of this type.

        Sprint 3: Data Validation Pipeline - checks previously approved facts
        to detect conflicts with new extractions.

        Args:
            person_id: ID of the person to query
            fact_type: Type of fact to query

        Returns:
            List of ExistingFact objects from approved SuggestedFact records
        """
        if not self.db_session:
            return []

        try:
            from core.database import FactStatusEnum, FactTypeEnum, SuggestedFact

            # Map string to enum
            try:
                fact_type_enum = FactTypeEnum(fact_type)
            except ValueError:
                logger.debug(f"Unknown fact type '{fact_type}', skipping DB query")
                return []

            # Query approved facts of this type for this person
            approved_facts = (
                self.db_session.query(SuggestedFact)
                .filter(
                    SuggestedFact.people_id == person_id,
                    SuggestedFact.fact_type == fact_type_enum,
                    SuggestedFact.status == FactStatusEnum.APPROVED,
                )
                .all()
            )

            existing: list[ExistingFact] = []
            for sf in approved_facts:
                existing.append(
                    ExistingFact(
                        fact_type=fact_type,
                        person_id=sf.people_id,
                        person_name=f"Person #{sf.people_id}",
                        value=sf.new_value,
                        source="SuggestedFact (Approved)",
                        last_updated=sf.updated_at,
                    )
                )

            if existing:
                logger.debug(f"Found {len(existing)} approved SuggestedFact records for person {person_id}")

            return existing

        except Exception as e:
            logger.warning(f"Error querying SuggestedFact table: {e}")
            return []

        return existing

    def _handle_no_existing(self, fact: ExtractedFact) -> ValidationResult:
        """Handle case where no existing fact exists."""
        if fact.confidence >= self.AUTO_APPROVE_MIN_CONFIDENCE:
            return ValidationResult(
                fact=fact,
                conflict_type=ConflictType.NO_EXISTING,
                auto_approved=True,
                review_priority=ReviewPriority.LOW,
                message=f"New fact with high confidence ({fact.confidence}%) - auto-approved",
            )
        return ValidationResult(
            fact=fact,
            conflict_type=ConflictType.NO_EXISTING,
            auto_approved=False,
            review_priority=ReviewPriority.NORMAL,
            message=f"New fact with moderate confidence ({fact.confidence}%) - queued for review",
        )

    def _detect_conflict(self, new: ExtractedFact, existing: ExistingFact) -> ConflictType:
        """Detect the type of conflict between new and existing facts."""
        if new.fact_type in {"BIRTH", "DEATH", "MARRIAGE"}:
            return self._compare_dates(existing.value, new.structured_value)
        if new.fact_type == "RELATIONSHIP":
            return self._compare_relationships(existing.value, new.structured_value)
        if new.fact_type == "LOCATION":
            return self._compare_locations(existing.value, new.structured_value)
        # Default: use string similarity
        return self._compare_strings(existing.value, new.structured_value)

    def _compare_dates(self, existing: str, new: str) -> ConflictType:
        """
        Compare date values with tolerance.

        - Exact: "1920" == "1920"
        - Compatible: "1920" can be enhanced to "1920-03-15"
        - Minor: 1-2 year difference
        - Major: > 5 year difference
        """
        # Extract years from both values
        existing_year = self._extract_year(existing)
        new_year = self._extract_year(new)

        if existing_year is None or new_year is None:
            # Can't compare, treat as compatible (new adds info)
            return ConflictType.COMPATIBLE

        if existing_year == new_year:
            # Check if new date is more specific
            if len(new) > len(existing):
                return ConflictType.COMPATIBLE
            return ConflictType.EXACT_MATCH

        diff = abs(existing_year - new_year)
        if diff <= self.MINOR_CONFLICT_YEAR_DIFF:
            return ConflictType.MINOR_CONFLICT
        if diff <= self.MAJOR_CONFLICT_YEAR_DIFF:
            return ConflictType.MINOR_CONFLICT  # Still minor for 3-5 years
        return ConflictType.MAJOR_CONFLICT

    @staticmethod
    def _compare_relationships(existing: str, new: str) -> ConflictType:
        """Compare relationship values."""
        existing_norm = existing.lower().strip()
        new_norm = new.lower().strip()

        if existing_norm == new_norm:
            return ConflictType.EXACT_MATCH

        # Check for compatible relationships (e.g., "father" vs "parent")
        compatible_pairs = [
            ({"father", "dad", "parent"}, {"father", "dad", "parent"}),
            ({"mother", "mom", "parent"}, {"mother", "mom", "parent"}),
            ({"wife", "spouse"}, {"wife", "spouse"}),
            ({"husband", "spouse"}, {"husband", "spouse"}),
        ]

        for group1, group2 in compatible_pairs:
            if existing_norm in group1 and new_norm in group2:
                return ConflictType.COMPATIBLE

        return ConflictType.MINOR_CONFLICT

    @staticmethod
    def _compare_locations(existing: str, new: str) -> ConflictType:
        """
        Compare locations with hierarchy awareness.

        - Exact: "Ohio, USA" == "Ohio, USA"
        - Compatible: "USA" can be refined to "Ohio, USA"
        - Minor: Adjacent locations or different specificity
        - Major: Different countries/regions
        """
        existing_norm = existing.lower().strip()
        new_norm = new.lower().strip()

        if existing_norm == new_norm:
            return ConflictType.EXACT_MATCH

        # Check if one contains the other (more specific)
        if existing_norm in new_norm or new_norm in existing_norm:
            return ConflictType.COMPATIBLE

        # Use string similarity for fuzzy matching
        similarity = SequenceMatcher(None, existing_norm, new_norm).ratio()
        if similarity >= 0.8:
            return ConflictType.MINOR_CONFLICT

        return ConflictType.MAJOR_CONFLICT

    def _compare_strings(self, existing: str, new: str) -> ConflictType:
        """Generic string comparison using similarity ratio."""
        existing_norm = existing.lower().strip()
        new_norm = new.lower().strip()

        if existing_norm == new_norm:
            return ConflictType.EXACT_MATCH

        similarity = SequenceMatcher(None, existing_norm, new_norm).ratio()

        if similarity >= self.NAME_SIMILARITY_THRESHOLD:
            return ConflictType.COMPATIBLE
        if similarity >= 0.6:
            return ConflictType.MINOR_CONFLICT
        return ConflictType.MAJOR_CONFLICT

    @staticmethod
    def _create_result(fact: ExtractedFact, conflict: ConflictType, existing: ExistingFact) -> ValidationResult:
        """Create ValidationResult based on conflict type."""
        if conflict == ConflictType.COMPATIBLE:
            return ValidationResult(
                fact=fact,
                conflict_type=conflict,
                conflicting_fact=existing,
                auto_approved=fact.confidence >= 90,
                review_priority=ReviewPriority.NORMAL,
                message="Compatible fact - suggests merge with existing",
            )
        if conflict == ConflictType.MINOR_CONFLICT:
            return ValidationResult(
                fact=fact,
                conflict_type=conflict,
                conflicting_fact=existing,
                auto_approved=False,
                review_priority=ReviewPriority.HIGH,
                message=f"Minor conflict detected: existing='{existing.value}', new='{fact.structured_value}'",
            )
        # MAJOR_CONFLICT
        return ValidationResult(
            fact=fact,
            conflict_type=conflict,
            conflicting_fact=existing,
            auto_approved=False,
            review_priority=ReviewPriority.CRITICAL,
            message=f"MAJOR conflict: existing='{existing.value}', new='{fact.structured_value}'",
        )

    @staticmethod
    def _extract_year(date_str: str) -> Optional[int]:
        """Extract year from a date string."""
        if not date_str:
            return None

        # Match 4-digit year anywhere in string
        match = re.search(r"\b(\d{4})\b", date_str)
        if match:
            return int(match.group(1))
        return None

    def save_fact_to_db(self, result: ValidationResult, person_id: int) -> Optional[int]:
        """
        Save a validated fact to the SuggestedFact table.

        Sprint 3: Data Validation Pipeline - creates SuggestedFact records
        with appropriate status based on conflict detection and auto-approval.

        Args:
            result: ValidationResult from validate_fact()
            person_id: Person ID to associate the fact with

        Returns:
            ID of created SuggestedFact record, or None if save failed
        """
        if not self.db_session:
            logger.warning("No database session - cannot save fact to DB")
            return None

        try:
            from core.database import FactStatusEnum, FactTypeEnum, SuggestedFact

            # Map fact type to enum
            try:
                fact_type_enum = FactTypeEnum(result.fact.fact_type)
            except ValueError:
                logger.warning(f"Unknown fact type '{result.fact.fact_type}', cannot save to DB")
                return None

            # Determine status based on validation result
            if result.auto_approved:
                status = FactStatusEnum.APPROVED
            elif result.conflict_type in {ConflictType.MAJOR_CONFLICT}:
                status = FactStatusEnum.REJECTED  # Major conflicts default to rejected
            else:
                status = FactStatusEnum.PENDING

            # Create SuggestedFact record
            suggested_fact = SuggestedFact(
                people_id=person_id,
                fact_type=fact_type_enum,
                original_value=result.fact.original_text,
                new_value=result.fact.structured_value,
                source_message_id=result.fact.source_conversation_id,
                status=status,
                confidence_score=result.fact.confidence,
            )

            self.db_session.add(suggested_fact)
            self.db_session.flush()  # Get the ID without committing

            logger.info(
                f"Saved SuggestedFact #{suggested_fact.id}: {result.fact.fact_type} "
                f"for person {person_id} with status {status.value}"
            )

            return suggested_fact.id

        except Exception as e:
            logger.error(f"Error saving fact to database: {e}", exc_info=True)
            return None

    def validate_and_save_facts(
        self, facts: list[ExtractedFact], person: Person, commit: bool = False
    ) -> list[ValidationResult]:
        """
        Validate multiple facts and save them to the database.

        Sprint 3: Batch fact validation with database persistence.

        Args:
            facts: List of ExtractedFact objects to validate
            person: Person to validate facts against
            commit: Whether to commit the transaction after saving

        Returns:
            List of ValidationResult objects
        """
        results: list[ValidationResult] = []

        for fact in facts:
            # Validate the fact
            result = self.validate_fact(fact, person)

            # Save to database
            fact_id = self.save_fact_to_db(result, person.id)
            result.suggested_fact_id = fact_id

            # Log conflicts for visibility
            if result.conflict_type in {ConflictType.MINOR_CONFLICT, ConflictType.MAJOR_CONFLICT}:
                logger.warning(
                    f"Conflict detected for {person.display_name or person.id}: "
                    f"{result.fact.fact_type} - {result.message}"
                )

            results.append(result)

        if commit and self.db_session:
            try:
                self.db_session.commit()
                logger.info(f"Committed {len(results)} fact validation results to database")
            except Exception as e:
                self.db_session.rollback()
                logger.error(f"Failed to commit fact validation results: {e}")

        return results

    @staticmethod
    def get_conflicts_summary(results: list[ValidationResult]) -> dict[str, Any]:
        """
        Generate a summary of conflicts from validation results.

        Args:
            results: List of ValidationResult objects

        Returns:
            Summary dict with conflict counts and details
        """
        summary: dict[str, Any] = {
            "total_facts": len(results),
            "auto_approved": 0,
            "pending_review": 0,
            "exact_matches": 0,
            "compatible": 0,
            "minor_conflicts": 0,
            "major_conflicts": 0,
            "no_existing": 0,
            "conflict_details": [],
        }

        for result in results:
            if result.auto_approved:
                summary["auto_approved"] += 1

            if result.conflict_type == ConflictType.EXACT_MATCH:
                summary["exact_matches"] += 1
            elif result.conflict_type == ConflictType.COMPATIBLE:
                summary["compatible"] += 1
            elif result.conflict_type == ConflictType.MINOR_CONFLICT:
                summary["minor_conflicts"] += 1
                summary["conflict_details"].append(
                    {
                        "type": "MINOR",
                        "fact_type": result.fact.fact_type,
                        "new_value": result.fact.structured_value,
                        "existing_value": result.conflicting_fact.value if result.conflicting_fact else None,
                        "message": result.message,
                    }
                )
            elif result.conflict_type == ConflictType.MAJOR_CONFLICT:
                summary["major_conflicts"] += 1
                summary["conflict_details"].append(
                    {
                        "type": "MAJOR",
                        "fact_type": result.fact.fact_type,
                        "new_value": result.fact.structured_value,
                        "existing_value": result.conflicting_fact.value if result.conflicting_fact else None,
                        "message": result.message,
                    }
                )
            else:
                summary["no_existing"] += 1

            if not result.auto_approved and result.conflict_type != ConflictType.EXACT_MATCH:
                summary["pending_review"] += 1

        return summary


# === FACT EXTRACTION UTILITIES ===


def extract_facts_from_ai_response(
    ai_response: dict[str, Any],
    conversation_id: Optional[str] = None,
    original_message: str = "",
) -> list[ExtractedFact]:
    """
    Convert AI extraction response to standardized ExtractedFact objects.

    This is the main integration point for Fact Extraction 2.0.

    Args:
        ai_response: Raw AI response dict with 'extracted_data' key
        conversation_id: Source conversation for traceability
        original_message: The original message text that was analyzed

    Returns:
        List of ExtractedFact objects
    """
    facts: list[ExtractedFact] = []

    extracted_data = ai_response.get("extracted_data", {})

    # Extract from vital_records
    for record in extracted_data.get("vital_records", []):
        try:
            fact = ExtractedFact.from_vital_record(
                person_name=record.get("person", "Unknown"),
                event_type=record.get("event_type", "OTHER"),
                date=record.get("date", ""),
                place=record.get("place", ""),
                certainty=record.get("certainty", "uncertain"),
                original_text=original_message,
                conversation_id=conversation_id,
            )
            facts.append(fact)
        except Exception as e:
            logger.warning(f"Failed to create fact from vital_record: {e}")

    # Extract from relationships
    for rel in extracted_data.get("relationships", []):
        try:
            fact = ExtractedFact.from_relationship(
                person1=rel.get("person1", "Unknown"),
                relationship=rel.get("relationship", "related"),
                person2=rel.get("person2", "Unknown"),
                original_text=original_message,
                conversation_id=conversation_id,
            )
            facts.append(fact)
        except Exception as e:
            logger.warning(f"Failed to create fact from relationship: {e}")

    # Extract from mentioned_people (create BIRTH facts for birth years)
    for person in extracted_data.get("mentioned_people", []):
        name = person.get("name", person.get("full_name", "Unknown"))

        # Birth year
        birth_year = person.get("birth_year")
        if birth_year:
            facts.append(
                ExtractedFact(
                    fact_type="BIRTH",
                    subject_name=name,
                    original_text=original_message,
                    structured_value=str(birth_year),
                    normalized_value=f"{birth_year}-01-01",
                    confidence=85,
                    source_conversation_id=conversation_id,
                    location=person.get("birth_place"),
                )
            )

        # Death year
        death_year = person.get("death_year")
        if death_year:
            facts.append(
                ExtractedFact(
                    fact_type="DEATH",
                    subject_name=name,
                    original_text=original_message,
                    structured_value=str(death_year),
                    normalized_value=f"{death_year}-01-01",
                    confidence=85,
                    source_conversation_id=conversation_id,
                    location=person.get("death_place"),
                )
            )

        # Relationship
        relationship = person.get("relationship")
        if relationship:
            facts.append(
                ExtractedFact(
                    fact_type="RELATIONSHIP",
                    subject_name=name,
                    original_text=original_message,
                    structured_value=relationship,
                    normalized_value=relationship.lower(),
                    confidence=80,
                    source_conversation_id=conversation_id,
                    related_person_name="User",  # The match is related to the user
                )
            )

    logger.info(f"Extracted {len(facts)} facts from AI response")
    return facts


# === TEST SUITE ===


def fact_validator_tests() -> bool:
    """Run tests for the FactValidator module."""
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Fact Validator", __name__)
        suite.start_suite()

        # Test ExtractedFact creation
        def test_extracted_fact_creation():
            fact = ExtractedFact(
                fact_type="BIRTH",
                subject_name="Charles Fetch",
                original_text="Charles was born in 1881 in Banff",
                structured_value="1881",
                normalized_value="1881-01-01",
                confidence=95,
            )
            assert fact.fact_type == "BIRTH"
            assert fact.confidence == 95
            assert "1881" in fact.normalized_value

        # Test from_vital_record factory
        def test_from_vital_record():
            fact = ExtractedFact.from_vital_record(
                person_name="Mary MacDonald",
                event_type="birth",
                date="1885",
                place="Aberdeen, Scotland",
                certainty="certain",
                original_text="Mary was born in 1885",
            )
            assert fact.fact_type == "BIRTH"
            assert fact.subject_name == "Mary MacDonald"
            assert fact.confidence == 95  # certain -> 95
            assert fact.location == "Aberdeen, Scotland"

        # Test date normalization
        def test_date_normalization():
            # Year only
            assert ExtractedFact._normalize_date("1920") == "1920-01-01"
            # Already ISO
            assert ExtractedFact._normalize_date("1920-03-15") == "1920-03-15"
            # Circa
            assert ExtractedFact._normalize_date("circa 1920") == "1920-01-01"
            assert ExtractedFact._normalize_date("~1920") == "1920-01-01"
            # Before/After
            assert ExtractedFact._normalize_date("before 1920") == "1920-01-01"

        # Test FactValidator conflict detection
        def test_conflict_detection_exact():
            validator = FactValidator()
            conflict = validator._compare_dates("1920", "1920")
            assert conflict == ConflictType.EXACT_MATCH

        def test_conflict_detection_minor():
            validator = FactValidator()
            conflict = validator._compare_dates("1920", "1921")
            assert conflict == ConflictType.MINOR_CONFLICT

        def test_conflict_detection_major():
            validator = FactValidator()
            conflict = validator._compare_dates("1920", "1850")
            assert conflict == ConflictType.MAJOR_CONFLICT

        def test_conflict_detection_compatible():
            validator = FactValidator()
            conflict = validator._compare_dates("1920", "1920-03-15")
            assert conflict == ConflictType.COMPATIBLE

        # Test extraction from AI response
        def test_extract_facts_from_ai():
            ai_response = {
                "extracted_data": {
                    "vital_records": [
                        {
                            "person": "Charles Fetch",
                            "event_type": "birth",
                            "date": "1881",
                            "place": "Banff, Scotland",
                            "certainty": "certain",
                        }
                    ],
                    "mentioned_people": [{"name": "Mary MacDonald", "birth_year": 1885, "relationship": "his wife"}],
                    "relationships": [
                        {
                            "person1": "Charles Fetch",
                            "relationship": "spouse",
                            "person2": "Mary MacDonald",
                            "context": "married 1908",
                        }
                    ],
                }
            }
            facts = extract_facts_from_ai_response(ai_response, "conv-123", "Test message")
            assert len(facts) >= 3  # 1 vital + 2 from mentioned (birth + relationship) + 1 relationship
            birth_facts = [f for f in facts if f.fact_type == "BIRTH"]
            assert len(birth_facts) >= 1

        # Test validation result
        def test_validation_no_existing():
            validator = FactValidator()
            fact = ExtractedFact(
                fact_type="BIRTH",
                subject_name="Test Person",
                original_text="Born 1920",
                structured_value="1920",
                normalized_value="1920-01-01",
                confidence=90,
            )
            result = validator.validate_fact(fact, person=None)
            assert result.conflict_type == ConflictType.NO_EXISTING
            assert result.auto_approved is True  # High confidence

        def test_validation_low_confidence():
            validator = FactValidator()
            fact = ExtractedFact(
                fact_type="BIRTH",
                subject_name="Test Person",
                original_text="Born around 1920",
                structured_value="1920",
                normalized_value="1920-01-01",
                confidence=50,  # Low confidence
            )
            result = validator.validate_fact(fact, person=None)
            assert result.auto_approved is False

        suite.run_test("ExtractedFact creation", test_extracted_fact_creation, "Should create fact")
        suite.run_test("from_vital_record", test_from_vital_record, "Factory method works")
        suite.run_test("Date normalization", test_date_normalization, "Dates normalized to ISO")
        suite.run_test("Conflict: exact match", test_conflict_detection_exact, "Same dates -> EXACT_MATCH")
        suite.run_test("Conflict: minor", test_conflict_detection_minor, "1-2 year diff -> MINOR")
        suite.run_test("Conflict: major", test_conflict_detection_major, ">5 year diff -> MAJOR")
        suite.run_test("Conflict: compatible", test_conflict_detection_compatible, "More specific -> COMPATIBLE")
        suite.run_test("Extract from AI response", test_extract_facts_from_ai, "Parse AI output to facts")
        suite.run_test("Validation: no existing", test_validation_no_existing, "High confidence auto-approves")
        suite.run_test("Validation: low confidence", test_validation_low_confidence, "Low confidence needs review")

        # Phase 3.1: New tests for from_conversation and date_qualifier
        def test_from_conversation():
            """Test from_conversation factory method."""
            extracted_data = {
                "vital_records": [
                    {"person": "John Smith", "event_type": "birth", "date": "circa 1920", "place": "Boston, MA", "certainty": "probable"},
                ],
                "relationships": [
                    {"person1": "John Smith", "relationship": "father", "person2": "Mary Smith"},
                ],
                "mentioned_people": [
                    {"name": "Jane Doe", "birth_year": 1950, "death_year": 2020, "birth_place": "New York"},
                ],
            }
            facts = ExtractedFact.from_conversation("conv-456", extracted_data, "Test message")
            assert len(facts) >= 4  # 1 vital + 1 relationship + 2 from mentioned_people (birth + death)
            # Check vital record was created
            birth_facts = [f for f in facts if f.fact_type == "BIRTH" and "John" in f.subject_name]
            assert len(birth_facts) == 1
            assert birth_facts[0].date_qualifier == "circa"
            # Check relationship was created
            rel_facts = [f for f in facts if f.fact_type == "RELATIONSHIP"]
            assert len(rel_facts) == 1
            assert rel_facts[0].related_person_name == "Mary Smith"
            # Check mentioned_people created facts
            jane_facts = [f for f in facts if "Jane" in f.subject_name]
            assert len(jane_facts) >= 2  # Birth and death

        def test_extract_date_qualifier():
            """Test _extract_date_qualifier static method."""
            assert ExtractedFact._extract_date_qualifier("circa 1920") == "circa"
            assert ExtractedFact._extract_date_qualifier("about 1920") == "circa"
            assert ExtractedFact._extract_date_qualifier("~1920") == "circa"
            assert ExtractedFact._extract_date_qualifier("c.1920") == "circa"
            assert ExtractedFact._extract_date_qualifier("before 1920") == "before"
            assert ExtractedFact._extract_date_qualifier("bef 1920") == "before"
            assert ExtractedFact._extract_date_qualifier("after 1920") == "after"
            assert ExtractedFact._extract_date_qualifier("aft. 1920") == "after"
            assert ExtractedFact._extract_date_qualifier("1920") is None
            assert ExtractedFact._extract_date_qualifier("") is None

        def test_from_vital_record_with_qualifier():
            """Test that from_vital_record sets date_qualifier."""
            fact = ExtractedFact.from_vital_record(
                person_name="John Smith",
                event_type="birth",
                date="circa 1920",
                place="Boston",
                certainty="probable",
                original_text="Born circa 1920",
            )
            assert fact.date_qualifier == "circa"
            assert fact.normalized_value == "1920-01-01"

        suite.run_test("from_conversation factory", test_from_conversation, "Creates facts from AI extraction output")
        suite.run_test("Date qualifier extraction", test_extract_date_qualifier, "Extracts circa/before/after")
        suite.run_test("Vital record with qualifier", test_from_vital_record_with_qualifier, "Sets date_qualifier field")

        return suite.finish_suite()


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(fact_validator_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
