"""
Conflict Detection Module.

Provides automated detection and management of data conflicts between
extracted values (from AI/conversation) and existing database records.
Supports a review workflow for human validation.

Features:
- Automated field-level comparison
- Conflict scoring based on similarity
- Batch conflict detection
- Resolution tracking
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional

from sqlalchemy.orm import Session

if __package__ in {None, ""}:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from core.database import ConflictSeverityEnum, ConflictStatusEnum, DataConflict, Person
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


class ConflictSeverity(Enum):
    """Severity levels for data conflicts (module-level, maps to ConflictSeverityEnum)."""

    LOW = "low"  # Minor difference (typo, formatting)
    MEDIUM = "medium"  # Significant difference requiring review
    HIGH = "high"  # Major contradiction (different dates/places)
    CRITICAL = "critical"  # Fundamental conflict (relationship change)


@dataclass
class FieldComparison:
    """Result of comparing a single field."""

    field_name: str
    existing_value: Optional[str]
    new_value: str
    is_conflict: bool
    similarity_score: float  # 0.0 to 1.0
    severity: ConflictSeverity
    notes: str = ""


@dataclass
class ConflictDetectionResult:
    """Result of conflict detection for a person."""

    person_id: int
    person_name: str
    conflicts: list[FieldComparison] = field(default_factory=list)
    total_fields_checked: int = 0
    conflicts_found: int = 0
    source: str = "conversation"
    source_message_id: Optional[int] = None

    @property
    def has_conflicts(self) -> bool:
        """Return True if any conflicts were detected."""
        return self.conflicts_found > 0

    @property
    def max_severity(self) -> Optional[ConflictSeverity]:
        """Return the highest severity among detected conflicts."""
        if not self.conflicts:
            return None
        severity_order = [
            ConflictSeverity.LOW,
            ConflictSeverity.MEDIUM,
            ConflictSeverity.HIGH,
            ConflictSeverity.CRITICAL,
        ]
        max_sev = ConflictSeverity.LOW
        for conflict in self.conflicts:
            if severity_order.index(conflict.severity) > severity_order.index(max_sev):
                max_sev = conflict.severity
        return max_sev


class ConflictDetector:
    """
    Detects and manages data conflicts between extracted and stored values.

    Compares newly extracted data against existing database records and
    creates DataConflict entries for human review when discrepancies are found.
    """

    # Fields to compare and their severity when conflicting
    COMPARABLE_FIELDS: ClassVar[dict[str, ConflictSeverity]] = {
        "birth_year": ConflictSeverity.HIGH,
        "first_name": ConflictSeverity.MEDIUM,
        "gender": ConflictSeverity.HIGH,
        "birth_place": ConflictSeverity.HIGH,
        "death_year": ConflictSeverity.HIGH,
        "relationship": ConflictSeverity.CRITICAL,
    }

    # Similarity threshold below which values are considered different
    SIMILARITY_THRESHOLD = 0.85

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """
        Initialize the conflict detector.

        Args:
            similarity_threshold: Minimum similarity score (0-1) for values
                                  to be considered equivalent
        """
        self.similarity_threshold = similarity_threshold

    def compare_values(
        self,
        field_name: str,
        existing: Optional[Any],
        new: Any,
    ) -> FieldComparison:
        """
        Compare existing and new values for a field.

        Args:
            field_name: Name of the field being compared
            existing: Current value in database (can be None)
            new: New value from extraction

        Returns:
            FieldComparison with conflict analysis
        """
        # Normalize values for comparison
        existing_str = self._normalize_value(existing)
        new_str = self._normalize_value(new)

        # If existing is empty/None, no conflict (new data)
        if not existing_str:
            return FieldComparison(
                field_name=field_name,
                existing_value=existing_str,
                new_value=new_str,
                is_conflict=False,
                similarity_score=1.0,
                severity=ConflictSeverity.LOW,
                notes="New data for empty field",
            )

        # If values match exactly
        if existing_str == new_str:
            return FieldComparison(
                field_name=field_name,
                existing_value=existing_str,
                new_value=new_str,
                is_conflict=False,
                similarity_score=1.0,
                severity=ConflictSeverity.LOW,
                notes="Values match",
            )

        # Calculate similarity
        similarity = self._calculate_similarity(existing_str, new_str)

        # Determine if it's a conflict based on threshold
        is_conflict = similarity < self.similarity_threshold

        # Get severity for this field
        severity = self.COMPARABLE_FIELDS.get(field_name, ConflictSeverity.MEDIUM)

        # Adjust severity based on similarity (closer values = less severe)
        if similarity > 0.7 and severity != ConflictSeverity.CRITICAL:
            severity = ConflictSeverity.LOW

        notes = ""
        if is_conflict:
            notes = f"Values differ (similarity: {similarity:.1%})"
        else:
            notes = f"Values similar enough (similarity: {similarity:.1%})"

        return FieldComparison(
            field_name=field_name,
            existing_value=existing_str,
            new_value=new_str,
            is_conflict=is_conflict,
            similarity_score=similarity,
            severity=severity,
            notes=notes,
        )

    def detect_conflicts(
        self,
        person: Person,
        extracted_data: dict[str, Any],
        source: str = "conversation",
        source_message_id: Optional[int] = None,
    ) -> ConflictDetectionResult:
        """
        Detect conflicts between person record and extracted data.

        Args:
            person: Existing Person record from database
            extracted_data: Dictionary of newly extracted field values
            source: Source of the extracted data
            source_message_id: Optional reference to source conversation

        Returns:
            ConflictDetectionResult with all detected conflicts
        """
        result = ConflictDetectionResult(
            person_id=person.id,
            person_name=person.username or f"Person #{person.id}",
            source=source,
            source_message_id=source_message_id,
        )

        # Compare each field that has a new value
        for field_name, new_value in extracted_data.items():
            if field_name not in self.COMPARABLE_FIELDS:
                continue

            if new_value is None:
                continue

            # Get existing value from person
            existing_value = getattr(person, field_name, None)

            # Compare
            comparison = self.compare_values(field_name, existing_value, new_value)
            result.total_fields_checked += 1

            if comparison.is_conflict:
                result.conflicts.append(comparison)
                result.conflicts_found += 1

        return result

    @staticmethod
    def create_conflict_records(
        db_session: Session,
        detection_result: ConflictDetectionResult,
        confidence_score: Optional[int] = None,
    ) -> list[DataConflict]:
        """
        Create DataConflict records in the database.

        Args:
            db_session: Active database session
            detection_result: Result from detect_conflicts
            confidence_score: Optional AI confidence score

        Returns:
            List of created DataConflict records
        """
        created_records: list[DataConflict] = []

        for conflict in detection_result.conflicts:
            # Check if a similar conflict already exists (OPEN status, same field)
            existing = (
                db_session.query(DataConflict)
                .filter(
                    DataConflict.people_id == detection_result.person_id,
                    DataConflict.field_name == conflict.field_name,
                    DataConflict.status == ConflictStatusEnum.OPEN,
                )
                .first()
            )

            if existing:
                # Update existing conflict with new value if different
                if existing.new_value != conflict.new_value:
                    existing.new_value = conflict.new_value or ""
                    existing.source_message_id = detection_result.source_message_id
                    existing.confidence_score = confidence_score
                    logger.debug(
                        f"Updated existing conflict for person {detection_result.person_id}, "
                        f"field '{conflict.field_name}'"
                    )
                continue

            # Create new conflict record
            conflict_record = DataConflict(
                people_id=detection_result.person_id,
                field_name=conflict.field_name,
                existing_value=conflict.existing_value,
                new_value=conflict.new_value or "",
                source=detection_result.source,
                source_message_id=detection_result.source_message_id,
                confidence_score=confidence_score,
                status=ConflictStatusEnum.OPEN,
            )
            db_session.add(conflict_record)
            created_records.append(conflict_record)

            logger.info(
                f"Created conflict record for person {detection_result.person_id}: "
                f"'{conflict.field_name}' ({conflict.existing_value} -> {conflict.new_value})"
            )

        return created_records

    @staticmethod
    def resolve_conflict(
        db_session: Session,
        conflict_id: int,
        resolution: ConflictStatusEnum,
        apply_new_value: bool = False,
        resolution_notes: Optional[str] = None,
        resolved_by: str = "user",
    ) -> Optional[DataConflict]:
        """
        Resolve a data conflict.

        Args:
            db_session: Active database session
            conflict_id: ID of the conflict to resolve
            resolution: Resolution status (RESOLVED, IGNORED, AUTO_RESOLVED)
            apply_new_value: If True, update the Person record with new value
            resolution_notes: Optional notes about the resolution
            resolved_by: Identifier of who/what resolved the conflict

        Returns:
            Updated DataConflict record, or None if not found
        """
        conflict = db_session.query(DataConflict).filter(DataConflict.id == conflict_id).first()

        if not conflict:
            logger.warning(f"Conflict {conflict_id} not found")
            return None

        # Update conflict status
        conflict.status = resolution
        conflict.resolution_notes = resolution_notes
        conflict.resolved_by = resolved_by
        conflict.resolved_at = datetime.now(timezone.utc)

        # Apply new value to person if requested
        if apply_new_value and resolution == ConflictStatusEnum.RESOLVED:
            person = db_session.query(Person).filter(Person.id == conflict.people_id).first()
            if person and hasattr(person, conflict.field_name):
                old_value = getattr(person, conflict.field_name)
                setattr(person, conflict.field_name, conflict.new_value)
                logger.info(
                    f"Applied conflict resolution for person {conflict.people_id}: "
                    f"'{conflict.field_name}' {old_value} -> {conflict.new_value}"
                )

        return conflict

    @staticmethod
    def get_open_conflicts(
        db_session: Session,
        person_id: Optional[int] = None,
        limit: int = 100,
    ) -> list[DataConflict]:
        """
        Get open conflicts, optionally filtered by person.

        Args:
            db_session: Active database session
            person_id: Optional person ID to filter by
            limit: Maximum number of conflicts to return

        Returns:
            List of open DataConflict records
        """
        query = db_session.query(DataConflict).filter(DataConflict.status == ConflictStatusEnum.OPEN)

        if person_id is not None:
            query = query.filter(DataConflict.people_id == person_id)

        return query.order_by(DataConflict.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_critical_conflicts(
        db_session: Session,
        limit: int = 50,
    ) -> list[DataConflict]:
        """
        Get HIGH and CRITICAL severity conflicts requiring review.

        Phase 11.2: Routes significant conflicts to human review queue.
        These conflicts have fundamental data contradictions that need
        human validation before applying changes.

        Args:
            db_session: Active database session
            limit: Maximum number of conflicts to return

        Returns:
            List of HIGH/CRITICAL DataConflict records ordered by severity then date
        """
        from sqlalchemy import case

        # Create severity ordering (CRITICAL first, then HIGH)
        severity_order = case(
            (DataConflict.severity == ConflictSeverityEnum.CRITICAL, 1),
            (DataConflict.severity == ConflictSeverityEnum.HIGH, 2),
            else_=3,
        )

        query = (
            db_session.query(DataConflict)
            .filter(
                DataConflict.status == ConflictStatusEnum.OPEN,
                DataConflict.severity.in_([ConflictSeverityEnum.HIGH, ConflictSeverityEnum.CRITICAL]),
            )
            .order_by(severity_order, DataConflict.created_at.desc())
            .limit(limit)
        )

        return query.all()

    @staticmethod
    def get_conflict_summary(
        db_session: Session,
    ) -> dict[str, Any]:
        """
        Get summary statistics for all conflicts.

        Args:
            db_session: Active database session

        Returns:
            Dictionary with conflict statistics
        """
        from sqlalchemy import func

        # Count by status
        status_counts = (
            db_session.query(DataConflict.status, func.count(DataConflict.id)).group_by(DataConflict.status).all()
        )

        # Count by field
        field_counts = (
            db_session.query(DataConflict.field_name, func.count(DataConflict.id))
            .filter(DataConflict.status == ConflictStatusEnum.OPEN)
            .group_by(DataConflict.field_name)
            .all()
        )

        # Convert Row tuples to dict - explicit for type safety
        field_dict: dict[str, int] = {}
        for field_name, count in field_counts:
            field_dict[field_name] = count

        return {
            "total_open": sum(c for s, c in status_counts if s == ConflictStatusEnum.OPEN),
            "total_resolved": sum(c for s, c in status_counts if s == ConflictStatusEnum.RESOLVED),
            "total_ignored": sum(c for s, c in status_counts if s == ConflictStatusEnum.IGNORED),
            "by_status": {s.value: c for s, c in status_counts},
            "by_field": field_dict,
        }

    @staticmethod
    def _normalize_value(value: Any) -> str:
        """Normalize a value for comparison."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(value)

    @staticmethod
    def _calculate_similarity(s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def _test_compare_values_exact_match() -> None:
    """Test that exact matches are not conflicts."""
    detector = ConflictDetector()
    result = detector.compare_values("birth_year", "1900", "1900")
    assert not result.is_conflict
    assert result.similarity_score == 1.0


def _test_compare_values_conflict() -> None:
    """Test that different values are conflicts."""
    detector = ConflictDetector()
    result = detector.compare_values("birth_year", "1900", "1885")
    assert result.is_conflict
    assert result.similarity_score < 0.85


def _test_compare_values_empty_existing() -> None:
    """Test that empty existing value is not a conflict."""
    detector = ConflictDetector()
    result = detector.compare_values("birth_place", None, "London")
    assert not result.is_conflict


def _test_compare_values_similar() -> None:
    """Test that similar values may not be conflicts."""
    detector = ConflictDetector()
    result = detector.compare_values("first_name", "John", "john")
    assert not result.is_conflict  # Case-insensitive match


def _test_normalize_value() -> None:
    """Test value normalization."""
    assert not ConflictDetector._normalize_value(None)
    assert ConflictDetector._normalize_value("  Test  ") == "test"
    assert ConflictDetector._normalize_value(1900) == "1900"


def _test_calculate_similarity() -> None:
    """Test similarity calculation."""
    sim = ConflictDetector._calculate_similarity
    assert sim("test", "test") == 1.0
    assert sim("test", "TEST") == 1.0
    assert sim("John", "Jon") > 0.7
    assert sim("London", "Paris") < 0.5


def _test_conflict_detection_result() -> None:
    """Test ConflictDetectionResult properties."""
    result = ConflictDetectionResult(
        person_id=1,
        person_name="Test",
        conflicts=[
            FieldComparison("birth_year", "1900", "1885", True, 0.5, ConflictSeverity.HIGH),
            FieldComparison("first_name", "John", "Jon", True, 0.75, ConflictSeverity.MEDIUM),
        ],
        total_fields_checked=2,
        conflicts_found=2,
    )
    assert result.has_conflicts
    assert result.max_severity == ConflictSeverity.HIGH


def _test_severity_order() -> None:
    """Test conflict severity ordering."""
    result = ConflictDetectionResult(
        person_id=1,
        person_name="Test",
        conflicts=[
            FieldComparison("f1", "a", "b", True, 0.5, ConflictSeverity.LOW),
            FieldComparison("f2", "c", "d", True, 0.5, ConflictSeverity.CRITICAL),
            FieldComparison("f3", "e", "f", True, 0.5, ConflictSeverity.MEDIUM),
        ],
        conflicts_found=3,
    )
    assert result.max_severity == ConflictSeverity.CRITICAL


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Conflict Detection", "research/conflict_detector.py")

    suite.run_test("Compare values - exact match", _test_compare_values_exact_match)
    suite.run_test("Compare values - conflict", _test_compare_values_conflict)
    suite.run_test("Compare values - empty existing", _test_compare_values_empty_existing)
    suite.run_test("Compare values - similar", _test_compare_values_similar)
    suite.run_test("Normalize value", _test_normalize_value)
    suite.run_test("Calculate similarity", _test_calculate_similarity)
    suite.run_test("Detection result properties", _test_conflict_detection_result)
    suite.run_test("Severity ordering", _test_severity_order)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
