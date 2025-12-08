"""
Predictive Gap Detection Module.

Identifies research gaps in family trees by analyzing patterns,
missing data, and potential connections. Uses heuristics and
statistical analysis to suggest high-value research targets.

Features:
- Identifies missing vital records (birth, death, marriage)
- Detects incomplete family units (missing spouses/children)
- Suggests DNA match research priorities
- Calculates research potential scores
"""

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of research gaps."""

    MISSING_BIRTH = "missing_birth"
    MISSING_DEATH = "missing_death"
    MISSING_MARRIAGE = "missing_marriage"
    MISSING_SPOUSE = "missing_spouse"
    MISSING_PARENTS = "missing_parents"
    MISSING_CHILDREN = "missing_children"
    INCOMPLETE_LOCATION = "incomplete_location"
    UNDOCUMENTED_SOURCE = "undocumented_source"
    DNA_MATCH_UNRESEARCHED = "dna_match_unresearched"
    BRICK_WALL = "brick_wall"


class Priority(Enum):
    """Research priority levels."""

    CRITICAL = "critical"  # Missing essential data
    HIGH = "high"  # Likely to yield results
    MEDIUM = "medium"  # Worth investigating
    LOW = "low"  # Optional improvement


@dataclass
class ResearchGap:
    """Represents a single research gap in the family tree."""

    gap_type: GapType
    person_id: Optional[str]
    person_name: str
    description: str
    priority: Priority
    potential_score: float  # 0-100
    suggested_actions: list[str]
    related_matches: list[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    notes: list[str] = field(default_factory=list)


@dataclass
class GapAnalysisReport:
    """Complete gap analysis report for a tree section."""

    total_gaps: int
    critical_gaps: int
    high_priority_gaps: int
    gaps: list[ResearchGap]
    top_recommendations: list[str]
    overall_completeness_score: float  # 0-100
    research_potential_score: float  # 0-100


class PredictiveGapDetector:
    """
    Analyzes family trees to identify research gaps and opportunities.

    Uses pattern analysis and heuristics to detect:
    - Missing vital records
    - Incomplete family structures
    - Unresearched DNA connections
    - Brick wall ancestors
    """

    # Weights for different gap types in scoring
    GAP_WEIGHTS: ClassVar[dict[GapType, float]] = {
        GapType.MISSING_PARENTS: 0.20,
        GapType.BRICK_WALL: 0.18,
        GapType.DNA_MATCH_UNRESEARCHED: 0.15,
        GapType.MISSING_BIRTH: 0.12,
        GapType.MISSING_DEATH: 0.10,
        GapType.MISSING_MARRIAGE: 0.08,
        GapType.MISSING_SPOUSE: 0.07,
        GapType.MISSING_CHILDREN: 0.05,
        GapType.INCOMPLETE_LOCATION: 0.03,
        GapType.UNDOCUMENTED_SOURCE: 0.02,
    }

    # Typical generation depth for analysis
    DEFAULT_GENERATION_DEPTH = 4

    def __init__(
        self,
        research_service: Optional[Any] = None,
        db_session: Optional[Any] = None,
    ):
        """
        Initialize the gap detector.

        Args:
            research_service: Service for GEDCOM/tree queries
            db_session: Database session for match data
        """
        self.research_service = research_service
        self.db_session = db_session

    def analyze_person(
        self,
        person_data: dict[str, Any],
        include_relatives: bool = True,
    ) -> list[ResearchGap]:
        """
        Analyze a single person for research gaps.

        Args:
            person_data: Dictionary with person information
            include_relatives: Whether to check for missing relatives

        Returns:
            List of identified research gaps
        """
        gaps: list[ResearchGap] = []
        person_name = person_data.get("name", "Unknown")
        person_id = person_data.get("id")

        # Check vital records
        gaps.extend(self._check_vital_records(person_data, person_id, person_name))

        # Check family structure
        if include_relatives:
            gaps.extend(self._check_family_structure(person_data, person_id, person_name))

        # Check location data
        gaps.extend(self._check_location_data(person_data, person_id, person_name))

        # Check source documentation
        gaps.extend(self._check_sources(person_data, person_id, person_name))

        return gaps

    def analyze_tree_section(
        self,
        root_person: dict[str, Any],
        generations: int = 4,
    ) -> GapAnalysisReport:
        """
        Analyze a section of the family tree for gaps.

        Args:
            root_person: Starting person for analysis
            generations: Number of generations to analyze

        Returns:
            Comprehensive gap analysis report
        """
        all_gaps: list[ResearchGap] = []

        # Analyze root person
        all_gaps.extend(self.analyze_person(root_person))

        # Analyze ancestors if available
        ancestors = root_person.get("ancestors", [])
        for ancestor in ancestors[: self._max_ancestors(generations)]:
            all_gaps.extend(self.analyze_person(ancestor, include_relatives=False))

        # Check for brick walls
        all_gaps.extend(self._detect_brick_walls(root_person, generations))

        # Sort by priority and score
        all_gaps.sort(key=lambda g: (g.priority.value, -g.potential_score))

        # Generate report
        return self._create_report(all_gaps)

    def identify_dna_research_priorities(
        self,
        matches: list[dict[str, Any]],
        max_results: int = 10,
    ) -> list[ResearchGap]:
        """
        Identify high-priority DNA matches for research.

        Args:
            matches: List of DNA match data
            max_results: Maximum number of priorities to return

        Returns:
            List of research gaps for DNA matches
        """
        gaps: list[ResearchGap] = []

        for match in matches:
            gap = self._evaluate_match_potential(match)
            if gap:
                gaps.append(gap)

        # Sort by potential score
        gaps.sort(key=lambda g: -g.potential_score)
        return gaps[:max_results]

    @staticmethod
    def suggest_research_actions(
        gap: ResearchGap,
    ) -> list[str]:
        """
        Generate specific research action suggestions for a gap.

        Args:
            gap: The research gap to address

        Returns:
            List of suggested research actions
        """
        suggestions: list[str] = []

        if gap.gap_type == GapType.MISSING_BIRTH:
            suggestions = [
                f"Search birth records for {gap.person_name}",
                "Check church baptism records",
                "Review census records for birth year estimates",
                "Look for family bible records",
            ]
        elif gap.gap_type == GapType.MISSING_DEATH:
            suggestions = [
                f"Search death records for {gap.person_name}",
                "Check cemetery records and FindAGrave",
                "Review obituary databases",
                "Look for probate records",
            ]
        elif gap.gap_type == GapType.MISSING_MARRIAGE:
            suggestions = [
                f"Search marriage records for {gap.person_name}",
                "Check church marriage registers",
                "Review newspaper announcements",
                "Look for marriage license applications",
            ]
        elif gap.gap_type == GapType.MISSING_PARENTS:
            suggestions = [
                f"Research parents of {gap.person_name}",
                "Check birth certificate for parent names",
                "Review census records for family groupings",
                "Analyze DNA matches for parental lines",
            ]
        elif gap.gap_type == GapType.DNA_MATCH_UNRESEARCHED:
            suggestions = [
                "Review match's family tree for common surnames",
                "Identify shared matches for clustering",
                "Send introductory message to match",
                "Research geographic connections",
            ]
        elif gap.gap_type == GapType.BRICK_WALL:
            suggestions = [
                "Review all available records for clues",
                "Analyze DNA matches beyond 4th cousin",
                "Research collateral lines (siblings)",
                "Check variant spellings of surname",
                "Expand geographic search area",
            ]
        else:
            suggestions = [
                f"Research {gap.gap_type.value} for {gap.person_name}",
                "Review existing records for clues",
                "Consult genealogical forums",
            ]

        return suggestions

    @staticmethod
    def calculate_completeness_score(
        person_data: dict[str, Any],
    ) -> float:
        """
        Calculate how complete a person's record is.

        Args:
            person_data: Dictionary with person information

        Returns:
            Completeness score from 0.0 to 100.0
        """
        score = 0.0
        max_score = 100.0

        # Check key fields and their weights
        field_weights = {
            "name": 15.0,
            "birth_date": 15.0,
            "birth_place": 10.0,
            "death_date": 10.0,
            "death_place": 5.0,
            "father": 15.0,
            "mother": 15.0,
            "spouse": 10.0,
            "sources": 5.0,
        }

        for field_name, weight in field_weights.items():
            if person_data.get(field_name):
                score += weight

        return min(max_score, score)

    def _check_vital_records(
        self,
        person_data: dict[str, Any],
        person_id: Optional[str],
        person_name: str,
    ) -> list[ResearchGap]:
        """Check for missing vital records."""
        gaps: list[ResearchGap] = []

        # Check birth information
        if not person_data.get("birth_date"):
            gaps.append(
                ResearchGap(
                    gap_type=GapType.MISSING_BIRTH,
                    person_id=person_id,
                    person_name=person_name,
                    description=f"Missing birth date for {person_name}",
                    priority=Priority.HIGH,
                    potential_score=75.0,
                    suggested_actions=self.suggest_research_actions(
                        ResearchGap(
                            gap_type=GapType.MISSING_BIRTH,
                            person_id=person_id,
                            person_name=person_name,
                            description="",
                            priority=Priority.HIGH,
                            potential_score=0,
                            suggested_actions=[],
                        )
                    ),
                )
            )

        # Check death information (if person is likely deceased)
        birth_year = person_data.get("birth_year", 0)
        is_likely_deceased = birth_year and birth_year < 1925
        if is_likely_deceased and not person_data.get("death_date"):
            gaps.append(
                ResearchGap(
                    gap_type=GapType.MISSING_DEATH,
                    person_id=person_id,
                    person_name=person_name,
                    description=f"Missing death date for {person_name} (born {birth_year})",
                    priority=Priority.MEDIUM,
                    potential_score=60.0,
                    suggested_actions=self.suggest_research_actions(
                        ResearchGap(
                            gap_type=GapType.MISSING_DEATH,
                            person_id=person_id,
                            person_name=person_name,
                            description="",
                            priority=Priority.MEDIUM,
                            potential_score=0,
                            suggested_actions=[],
                        )
                    ),
                )
            )

        return gaps

    def _check_family_structure(
        self,
        person_data: dict[str, Any],
        person_id: Optional[str],
        person_name: str,
    ) -> list[ResearchGap]:
        """Check for incomplete family structure."""
        gaps: list[ResearchGap] = []

        # Check for missing parents
        has_father = bool(person_data.get("father") or person_data.get("father_id"))
        has_mother = bool(person_data.get("mother") or person_data.get("mother_id"))

        if not has_father and not has_mother:
            gaps.append(
                ResearchGap(
                    gap_type=GapType.MISSING_PARENTS,
                    person_id=person_id,
                    person_name=person_name,
                    description=f"Both parents unknown for {person_name}",
                    priority=Priority.CRITICAL,
                    potential_score=90.0,
                    suggested_actions=self.suggest_research_actions(
                        ResearchGap(
                            gap_type=GapType.MISSING_PARENTS,
                            person_id=person_id,
                            person_name=person_name,
                            description="",
                            priority=Priority.CRITICAL,
                            potential_score=0,
                            suggested_actions=[],
                        )
                    ),
                )
            )
        elif not has_father or not has_mother:
            missing = "father" if not has_father else "mother"
            gaps.append(
                ResearchGap(
                    gap_type=GapType.MISSING_PARENTS,
                    person_id=person_id,
                    person_name=person_name,
                    description=f"Missing {missing} for {person_name}",
                    priority=Priority.HIGH,
                    potential_score=80.0,
                    suggested_actions=[f"Research {missing} of {person_name}"],
                )
            )

        return gaps

    @staticmethod
    def _check_location_data(
        person_data: dict[str, Any],
        person_id: Optional[str],
        person_name: str,
    ) -> list[ResearchGap]:
        """Check for incomplete location data."""
        gaps: list[ResearchGap] = []

        birth_place = person_data.get("birth_place", "")
        death_place = person_data.get("death_place", "")

        # Check if locations are too vague (just country or state)
        if birth_place and len(birth_place.split(",")) < 2:
            gaps.append(
                ResearchGap(
                    gap_type=GapType.INCOMPLETE_LOCATION,
                    person_id=person_id,
                    person_name=person_name,
                    description=f"Birth location too vague for {person_name}: {birth_place}",
                    priority=Priority.LOW,
                    potential_score=30.0,
                    suggested_actions=["Research specific birth location"],
                )
            )

        if death_place and len(death_place.split(",")) < 2:
            gaps.append(
                ResearchGap(
                    gap_type=GapType.INCOMPLETE_LOCATION,
                    person_id=person_id,
                    person_name=person_name,
                    description=f"Death location too vague for {person_name}: {death_place}",
                    priority=Priority.LOW,
                    potential_score=25.0,
                    suggested_actions=["Research specific death location"],
                )
            )

        return gaps

    @staticmethod
    def _check_sources(
        person_data: dict[str, Any],
        person_id: Optional[str],
        person_name: str,
    ) -> list[ResearchGap]:
        """Check for undocumented sources."""
        gaps: list[ResearchGap] = []

        sources = person_data.get("sources", [])
        if not sources:
            gaps.append(
                ResearchGap(
                    gap_type=GapType.UNDOCUMENTED_SOURCE,
                    person_id=person_id,
                    person_name=person_name,
                    description=f"No sources documented for {person_name}",
                    priority=Priority.LOW,
                    potential_score=20.0,
                    suggested_actions=["Add source citations for existing data"],
                )
            )

        return gaps

    def _detect_brick_walls(
        self,
        root_person: dict[str, Any],
        generations: int,
    ) -> list[ResearchGap]:
        """Detect brick wall ancestors."""
        gaps: list[ResearchGap] = []

        # Check ancestor lines that end abruptly
        ancestors = root_person.get("ancestors", [])
        for ancestor in ancestors:
            gen = ancestor.get("generation", 0)
            has_parents = bool(
                ancestor.get("father")
                or ancestor.get("mother")
                or ancestor.get("father_id")
                or ancestor.get("mother_id")
            )

            # If ancestor is within analysis range but has no parents
            if gen < generations and not has_parents:
                name = ancestor.get("name", "Unknown")
                gaps.append(
                    ResearchGap(
                        gap_type=GapType.BRICK_WALL,
                        person_id=ancestor.get("id"),
                        person_name=name,
                        description=f"Brick wall: Cannot trace beyond {name} (generation {gen})",
                        priority=Priority.CRITICAL,
                        potential_score=95.0,
                        suggested_actions=self.suggest_research_actions(
                            ResearchGap(
                                gap_type=GapType.BRICK_WALL,
                                person_id=ancestor.get("id"),
                                person_name=name,
                                description="",
                                priority=Priority.CRITICAL,
                                potential_score=0,
                                suggested_actions=[],
                            )
                        ),
                        estimated_effort="high",
                    )
                )

        return gaps

    @staticmethod
    def _calculate_cm_potential(shared_cm: float) -> float:
        """Calculate potential score contribution from shared cM."""
        if shared_cm >= 200:
            return 40.0
        if shared_cm >= 90:
            return 30.0
        if shared_cm >= 40:
            return 20.0
        return 10.0

    @staticmethod
    def _calculate_tree_potential(has_tree: bool, tree_size: int) -> float:
        """Calculate potential score contribution from tree availability."""
        if not has_tree:
            return 0.0
        bonus = 20.0
        if tree_size >= 500:
            bonus += 15.0
        elif tree_size >= 100:
            bonus += 10.0
        return bonus

    @staticmethod
    def _determine_priority(shared_cm: float) -> Priority:
        """Determine priority based on shared cM."""
        if shared_cm >= 200:
            return Priority.CRITICAL
        if shared_cm >= 90:
            return Priority.HIGH
        if shared_cm >= 40:
            return Priority.MEDIUM
        return Priority.LOW

    def _evaluate_match_potential(
        self,
        match: dict[str, Any],
    ) -> Optional[ResearchGap]:
        """Evaluate a DNA match for research potential."""
        shared_cm = match.get("shared_cm", 0)
        has_tree = match.get("has_tree", False)
        tree_size = match.get("tree_size", 0)
        is_researched = match.get("is_researched", False)

        if is_researched:
            return None

        # Calculate potential score using helpers
        potential = self._calculate_cm_potential(shared_cm) + self._calculate_tree_potential(has_tree, tree_size)
        priority = self._determine_priority(shared_cm)
        match_name = match.get("name", "Unknown Match")

        return ResearchGap(
            gap_type=GapType.DNA_MATCH_UNRESEARCHED,
            person_id=match.get("uuid"),
            person_name=match_name,
            description=f"Unresearched DNA match: {match_name} ({shared_cm} cM)",
            priority=priority,
            potential_score=min(100.0, potential),
            suggested_actions=self.suggest_research_actions(
                ResearchGap(
                    gap_type=GapType.DNA_MATCH_UNRESEARCHED,
                    person_id=match.get("uuid"),
                    person_name=match_name,
                    description="",
                    priority=priority,
                    potential_score=0,
                    suggested_actions=[],
                )
            ),
            related_matches=match.get("shared_matches", []),
        )

    @staticmethod
    def _create_report(
        gaps: list[ResearchGap],
    ) -> GapAnalysisReport:
        """Create a gap analysis report from identified gaps."""
        critical_count = sum(1 for g in gaps if g.priority == Priority.CRITICAL)
        high_count = sum(1 for g in gaps if g.priority == Priority.HIGH)

        # Calculate overall scores
        total_potential = sum(g.potential_score for g in gaps)
        avg_potential = total_potential / len(gaps) if gaps else 0

        # Generate top recommendations
        top_recs: list[str] = []
        for gap in gaps[:5]:
            if gap.suggested_actions:
                top_recs.append(gap.suggested_actions[0])

        # Calculate completeness (inverse of gap severity)
        completeness = max(0, 100 - (critical_count * 15 + high_count * 10))

        return GapAnalysisReport(
            total_gaps=len(gaps),
            critical_gaps=critical_count,
            high_priority_gaps=high_count,
            gaps=gaps,
            top_recommendations=top_recs,
            overall_completeness_score=completeness,
            research_potential_score=min(100, avg_potential),
        )

    @staticmethod
    def _max_ancestors(generations: int) -> int:
        """Calculate maximum ancestors for given generations."""
        # 2^generations - 1 ancestors total
        return 2**generations - 1


# --- Test Suite ---


def _test_gap_types() -> None:
    """Test gap type enumeration."""
    assert GapType.MISSING_BIRTH.value == "missing_birth"
    assert GapType.BRICK_WALL.value == "brick_wall"
    assert len(GapType) == 10


def _test_priority_levels() -> None:
    """Test priority enumeration."""
    assert Priority.CRITICAL.value == "critical"
    assert Priority.LOW.value == "low"
    assert len(Priority) == 4


def _test_completeness_score() -> None:
    """Test completeness score calculation."""
    detector = PredictiveGapDetector()

    # Empty person
    score = detector.calculate_completeness_score({})
    assert score == 0.0

    # Complete person
    complete_person = {
        "name": "John Smith",
        "birth_date": "1900-01-01",
        "birth_place": "New York, USA",
        "death_date": "1980-12-31",
        "death_place": "Boston, MA",
        "father": "James Smith",
        "mother": "Mary Jones",
        "spouse": "Jane Doe",
        "sources": ["Census 1920"],
    }
    score = detector.calculate_completeness_score(complete_person)
    assert score == 100.0

    # Partial person
    partial = {"name": "John Smith", "birth_date": "1900"}
    score = detector.calculate_completeness_score(partial)
    assert 25.0 <= score <= 35.0


def _test_vital_record_gaps() -> None:
    """Test detection of missing vital records."""
    detector = PredictiveGapDetector()

    person = {
        "name": "John Smith",
        "birth_year": 1880,  # No birth_date
        # No death_date (likely deceased)
    }

    gaps = detector._check_vital_records(person, "id-1", "John Smith")

    assert len(gaps) == 2
    assert any(g.gap_type == GapType.MISSING_BIRTH for g in gaps)
    assert any(g.gap_type == GapType.MISSING_DEATH for g in gaps)


def _test_family_structure_gaps() -> None:
    """Test detection of missing family members."""
    detector = PredictiveGapDetector()

    # No parents
    person = {"name": "John Smith"}
    gaps = detector._check_family_structure(person, "id-1", "John Smith")

    assert len(gaps) == 1
    assert gaps[0].gap_type == GapType.MISSING_PARENTS
    assert gaps[0].priority == Priority.CRITICAL

    # One parent
    person_with_father = {"name": "John Smith", "father": "James Smith"}
    gaps = detector._check_family_structure(person_with_father, "id-1", "John Smith")

    assert len(gaps) == 1
    assert "mother" in gaps[0].description.lower()


def _test_dna_match_evaluation() -> None:
    """Test DNA match potential evaluation."""
    detector = PredictiveGapDetector()

    # High-value match
    match = {
        "name": "Jane Doe",
        "uuid": "match-1",
        "shared_cm": 250,
        "has_tree": True,
        "tree_size": 600,
        "is_researched": False,
    }

    gap = detector._evaluate_match_potential(match)

    assert gap is not None
    assert gap.gap_type == GapType.DNA_MATCH_UNRESEARCHED
    assert gap.priority == Priority.CRITICAL
    assert gap.potential_score >= 70

    # Already researched match
    researched = {"shared_cm": 200, "is_researched": True}
    gap = detector._evaluate_match_potential(researched)
    assert gap is None


def _test_research_actions() -> None:
    """Test research action suggestions."""
    detector = PredictiveGapDetector()

    birth_gap = ResearchGap(
        gap_type=GapType.MISSING_BIRTH,
        person_id="id-1",
        person_name="John Smith",
        description="",
        priority=Priority.HIGH,
        potential_score=75.0,
        suggested_actions=[],
    )

    actions = detector.suggest_research_actions(birth_gap)

    assert len(actions) > 0
    assert any("birth" in a.lower() for a in actions)


def _test_brick_wall_detection() -> None:
    """Test brick wall ancestor detection."""
    detector = PredictiveGapDetector()

    root = {
        "name": "Me",
        "ancestors": [
            {"name": "Dad", "generation": 1, "father": "Grandpa"},
            {"name": "Grandpa", "generation": 2},  # No parents = brick wall
        ],
    }

    gaps = detector._detect_brick_walls(root, generations=4)

    assert len(gaps) == 1
    assert gaps[0].gap_type == GapType.BRICK_WALL
    assert "Grandpa" in gaps[0].person_name


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Predictive Gap Detection", "predictive_gaps.py")
    suite.start_suite()

    suite.run_test("Gap type enumeration", _test_gap_types)
    suite.run_test("Priority level enumeration", _test_priority_levels)
    suite.run_test("Completeness score calculation", _test_completeness_score)
    suite.run_test("Vital record gap detection", _test_vital_record_gaps)
    suite.run_test("Family structure gap detection", _test_family_structure_gaps)
    suite.run_test("DNA match evaluation", _test_dna_match_evaluation)
    suite.run_test("Research action suggestions", _test_research_actions)
    suite.run_test("Brick wall detection", _test_brick_wall_detection)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
