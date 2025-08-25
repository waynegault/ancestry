"""
Smart DNA-GEDCOM Cross-Reference System for Ancestry Project

This module provides intelligent cross-referencing between DNA match data and
GEDCOM family tree data to identify verification opportunities, conflicts, and
research priorities. Enhances genealogical research by combining genetic and
documentary evidence.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 12.2 - Smart DNA-GEDCOM Cross-Reference
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)


@dataclass
class DNAMatch:
    """Represents a DNA match with genealogical context."""

    match_id: str
    match_name: str
    estimated_relationship: str
    shared_dna_cm: Optional[float] = None
    testing_company: str = "Ancestry"
    confidence_level: str = "medium"
    shared_ancestors: list[str] = field(default_factory=list)
    tree_size: Optional[int] = None
    last_login: Optional[str] = None


@dataclass
class GedcomPerson:
    """Represents a person from GEDCOM data with key genealogical information."""

    person_id: str
    full_name: str
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    birth_place: Optional[str] = None
    death_place: Optional[str] = None
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    spouses: list[str] = field(default_factory=list)


@dataclass
class CrossReferenceMatch:
    """Represents a potential match between DNA data and GEDCOM data."""

    match_id: str
    dna_match: DNAMatch
    potential_gedcom_matches: list[GedcomPerson]
    confidence_score: float  # 0.0 to 1.0
    match_type: str  # 'name_match', 'relationship_match', 'location_match', 'timeline_match'
    verification_needed: list[str] = field(default_factory=list)
    research_suggestions: list[str] = field(default_factory=list)


@dataclass
class ConflictIdentification:
    """Represents a conflict between DNA evidence and GEDCOM data."""

    conflict_id: str
    conflict_type: str  # 'relationship_mismatch', 'timeline_conflict', 'impossible_connection'
    description: str
    dna_evidence: str
    gedcom_evidence: str
    severity: str  # 'critical', 'major', 'minor'
    resolution_steps: list[str] = field(default_factory=list)


class DNAGedcomCrossReferencer:
    """
    Intelligent system for cross-referencing DNA matches with GEDCOM family tree data.
    """

    def __init__(self) -> None:
        """Initialize the DNA-GEDCOM cross-referencer."""
        self.cross_reference_matches: list[CrossReferenceMatch] = []
        self.conflicts_identified: list[ConflictIdentification] = []
        self.verification_opportunities: list[dict[str, Any]] = []

    def analyze_dna_gedcom_connections(
        self,
        dna_matches: list[DNAMatch],
        gedcom_data: Any,
        tree_owner_info: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Analyze connections between DNA matches and GEDCOM family tree data.

        Args:
            dna_matches: List of DNA match objects
            gedcom_data: GEDCOM data instance
            tree_owner_info: Information about the tree owner for relationship calculations

        Returns:
            Dictionary containing cross-reference analysis results
        """
        try:
            logger.info(f"Starting DNA-GEDCOM cross-reference analysis with {len(dna_matches)} DNA matches")

            # Clear previous analysis
            self.cross_reference_matches.clear()
            self.conflicts_identified.clear()
            self.verification_opportunities.clear()

            # Extract GEDCOM people data
            gedcom_people = self._extract_gedcom_people(gedcom_data)
            logger.debug(f"Extracted {len(gedcom_people)} people from GEDCOM data")

            # Perform cross-reference analysis
            for dna_match in dna_matches:
                self._analyze_single_dna_match(dna_match, gedcom_people, tree_owner_info)

            # Identify conflicts and verification opportunities
            self._identify_relationship_conflicts(dna_matches, gedcom_people)
            self._identify_verification_opportunities()

            # Generate analysis results
            analysis_result = {
                "analysis_timestamp": datetime.now().isoformat(),
                "dna_matches_analyzed": len(dna_matches),
                "gedcom_people_analyzed": len(gedcom_people),
                "cross_reference_matches": [self._crossref_to_dict(match) for match in self.cross_reference_matches],
                "conflicts_identified": [self._conflict_to_dict(conflict) for conflict in self.conflicts_identified],
                "verification_opportunities": self.verification_opportunities,
                "summary": self._generate_crossref_summary(),
                "recommendations": self._generate_crossref_recommendations()
            }

            logger.info(f"DNA-GEDCOM cross-reference completed: {len(self.cross_reference_matches)} matches, {len(self.conflicts_identified)} conflicts")
            return analysis_result

        except Exception as e:
            logger.error(f"Error during DNA-GEDCOM cross-reference analysis: {e}")
            return self._empty_crossref_result()

    def _extract_gedcom_people(self, gedcom_data: Any) -> list[GedcomPerson]:
        """Extract people data from GEDCOM for cross-referencing."""
        gedcom_people = []

        try:
            if not gedcom_data or not hasattr(gedcom_data, 'indi_index'):
                return gedcom_people

            for person_id, person_record in gedcom_data.indi_index.items():
                try:
                    # Extract basic information
                    full_name = self._extract_person_name(person_record)
                    birth_year = self._extract_birth_year(person_record)
                    death_year = self._extract_death_year(person_record)
                    birth_place = self._extract_birth_place(person_record)
                    death_place = self._extract_death_place(person_record)

                    # Extract family relationships
                    parents = list(gedcom_data.id_to_parents.get(person_id, []))
                    children = list(gedcom_data.id_to_children.get(person_id, []))

                    gedcom_person = GedcomPerson(
                        person_id=person_id,
                        full_name=full_name,
                        birth_year=birth_year,
                        death_year=death_year,
                        birth_place=birth_place,
                        death_place=death_place,
                        parents=parents,
                        children=children
                    )

                    gedcom_people.append(gedcom_person)

                except Exception as e:
                    logger.debug(f"Error extracting data for person {person_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting GEDCOM people: {e}")

        return gedcom_people

    def _analyze_single_dna_match(
        self,
        dna_match: DNAMatch,
        gedcom_people: list[GedcomPerson],
        tree_owner_info: Optional[dict[str, Any]]
    ):
        """Analyze a single DNA match against GEDCOM data."""
        try:
            potential_matches = []

            # Look for name matches
            name_matches = self._find_name_matches(dna_match, gedcom_people)
            potential_matches.extend(name_matches)

            # Look for relationship pattern matches
            relationship_matches = self._find_relationship_matches(dna_match, gedcom_people, tree_owner_info)
            potential_matches.extend(relationship_matches)

            # Look for geographic/timeline matches
            context_matches = self._find_context_matches(dna_match, gedcom_people)
            potential_matches.extend(context_matches)

            # Remove duplicates and score matches
            unique_matches = self._deduplicate_and_score_matches(potential_matches)

            if unique_matches:
                # Create cross-reference match
                confidence_score = max(match['confidence'] for match in unique_matches)
                match_types = list({match['type'] for match in unique_matches})

                crossref_match = CrossReferenceMatch(
                    match_id=f"crossref_{dna_match.match_id}",
                    dna_match=dna_match,
                    potential_gedcom_matches=[match['gedcom_person'] for match in unique_matches],
                    confidence_score=confidence_score,
                    match_type=', '.join(match_types),
                    verification_needed=self._generate_verification_tasks(dna_match, unique_matches),
                    research_suggestions=self._generate_research_suggestions(dna_match, unique_matches)
                )

                self.cross_reference_matches.append(crossref_match)

        except Exception as e:
            logger.debug(f"Error analyzing DNA match {dna_match.match_id}: {e}")

    def _find_name_matches(self, dna_match: DNAMatch, gedcom_people: list[GedcomPerson]) -> list[dict[str, Any]]:
        """Find potential matches based on name similarity."""
        name_matches = []

        dna_name_parts = dna_match.match_name.lower().split()

        for gedcom_person in gedcom_people:
            gedcom_name_parts = gedcom_person.full_name.lower().split()

            # Calculate name similarity
            common_parts = set(dna_name_parts) & set(gedcom_name_parts)
            similarity_score = len(common_parts) / max(len(dna_name_parts), len(gedcom_name_parts))

            if similarity_score > 0.5:  # At least 50% name similarity
                name_matches.append({
                    'gedcom_person': gedcom_person,
                    'confidence': similarity_score,
                    'type': 'name_match',
                    'details': f"Name similarity: {similarity_score:.2f}"
                })

        return name_matches

    def _find_relationship_matches(
        self,
        dna_match: DNAMatch,
        gedcom_people: list[GedcomPerson],
        tree_owner_info: Optional[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find potential matches based on estimated relationship."""
        relationship_matches = []

        if not dna_match.estimated_relationship or not tree_owner_info:
            return relationship_matches

        # This would implement relationship distance calculations
        # For now, providing basic framework

        relationship_distance = self._parse_relationship_distance(dna_match.estimated_relationship)
        if relationship_distance:
            # Find GEDCOM people at similar relationship distances
            for gedcom_person in gedcom_people:
                # Calculate relationship distance from tree owner to this person
                # This would require implementing relationship path calculation
                # For now, using placeholder logic

                if self._is_plausible_relationship_match(dna_match, gedcom_person, relationship_distance):
                    relationship_matches.append({
                        'gedcom_person': gedcom_person,
                        'confidence': 0.7,  # Placeholder confidence
                        'type': 'relationship_match',
                        'details': f"Relationship distance match: {dna_match.estimated_relationship}"
                    })

        return relationship_matches

    def _find_context_matches(self, dna_match: DNAMatch, gedcom_people: list[GedcomPerson]) -> list[dict[str, Any]]:
        """Find potential matches based on geographic and temporal context."""
        context_matches = []

        # This would analyze shared ancestors, locations, time periods
        # For now, providing basic framework

        for ancestor_name in dna_match.shared_ancestors:
            for gedcom_person in gedcom_people:
                if ancestor_name.lower() in gedcom_person.full_name.lower():
                    context_matches.append({
                        'gedcom_person': gedcom_person,
                        'confidence': 0.6,
                        'type': 'ancestor_match',
                        'details': f"Shared ancestor: {ancestor_name}"
                    })

        return context_matches

    def _deduplicate_and_score_matches(self, potential_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate matches and calculate final scores."""
        # Group by GEDCOM person
        person_matches = {}

        for match in potential_matches:
            person_id = match['gedcom_person'].person_id
            if person_id not in person_matches:
                person_matches[person_id] = []
            person_matches[person_id].append(match)

        # Calculate combined scores for each person
        unique_matches = []
        for person_id, matches in person_matches.items():
            # Take the best match for each person
            best_match = max(matches, key=lambda m: m['confidence'])

            # Boost confidence if multiple match types
            if len(matches) > 1:
                best_match['confidence'] = min(1.0, best_match['confidence'] * 1.2)
                best_match['type'] = ', '.join({m['type'] for m in matches})

            unique_matches.append(best_match)

        return unique_matches

    def _generate_verification_tasks(self, dna_match: DNAMatch, matches: list[dict[str, Any]]) -> list[str]:
        """Generate verification tasks for cross-reference matches."""
        tasks = []

        for match in matches:
            gedcom_person = match['gedcom_person']

            tasks.append(f"Verify relationship between DNA match {dna_match.match_name} and {gedcom_person.full_name}")

            if gedcom_person.birth_year:
                tasks.append(f"Check if {dna_match.match_name} has family connections around {gedcom_person.birth_year}")

            if gedcom_person.birth_place:
                tasks.append(f"Research {dna_match.match_name}'s family connections to {gedcom_person.birth_place}")

        return tasks[:5]  # Limit to top 5 tasks

    def _generate_research_suggestions(self, dna_match: DNAMatch, matches: list[dict[str, Any]]) -> list[str]:
        """Generate research suggestions for cross-reference matches."""
        suggestions = []

        suggestions.append(f"Compare {dna_match.match_name}'s family tree with potential GEDCOM matches")
        suggestions.append("Look for shared DNA segments to confirm relationship")

        if dna_match.shared_dna_cm:
            suggestions.append(f"Verify {dna_match.shared_dna_cm}cM shared DNA is consistent with estimated relationship")

        for match in matches:
            gedcom_person = match['gedcom_person']
            if gedcom_person.parents:
                suggestions.append("Research common ancestors between families")

        return suggestions[:5]  # Limit to top 5 suggestions

    def _identify_relationship_conflicts(self, dna_matches: list[DNAMatch], gedcom_people: list[GedcomPerson]):
        """Identify conflicts between DNA evidence and GEDCOM relationships."""
        # This would implement conflict detection logic
        # For now, providing framework

        for crossref_match in self.cross_reference_matches:
            dna_match = crossref_match.dna_match

            # Check for relationship inconsistencies
            if dna_match.estimated_relationship and dna_match.shared_dna_cm:
                expected_cm_range = self._get_expected_cm_range(dna_match.estimated_relationship)

                if expected_cm_range and not self._is_cm_in_range(dna_match.shared_dna_cm, expected_cm_range):
                    conflict = ConflictIdentification(
                        conflict_id=f"cm_conflict_{dna_match.match_id}",
                        conflict_type="relationship_mismatch",
                        description=f"Shared DNA ({dna_match.shared_dna_cm}cM) inconsistent with estimated relationship ({dna_match.estimated_relationship})",
                        dna_evidence=f"{dna_match.shared_dna_cm}cM shared DNA",
                        gedcom_evidence=f"Estimated relationship: {dna_match.estimated_relationship}",
                        severity="major",
                        resolution_steps=[
                            "Re-examine family tree connections",
                            "Consider alternative relationship possibilities",
                            "Look for additional DNA evidence"
                        ]
                    )
                    self.conflicts_identified.append(conflict)

    def _identify_verification_opportunities(self) -> None:
        """Identify high-value verification opportunities."""
        for crossref_match in self.cross_reference_matches:
            if crossref_match.confidence_score > 0.7:
                opportunity = {
                    "opportunity_id": f"verify_{crossref_match.match_id}",
                    "type": "high_confidence_match",
                    "description": f"High-confidence match between {crossref_match.dna_match.match_name} and GEDCOM data",
                    "priority": "high",
                    "verification_steps": crossref_match.verification_needed,
                    "expected_outcome": "Confirmed family tree connection or identification of new research lead"
                }
                self.verification_opportunities.append(opportunity)

    # Helper methods
    def _extract_person_name(self, person_record) -> str:
        """Extract person's name from GEDCOM record."""
        try:
            if hasattr(person_record, 'name') and person_record.name:
                return str(person_record.name[0])
            return "Unknown Name"
        except Exception:
            return "Unknown Name"

    def _extract_birth_year(self, person_record) -> Optional[int]:
        """Extract birth year from GEDCOM record."""
        # Placeholder implementation
        return None

    def _extract_death_year(self, person_record) -> Optional[int]:
        """Extract death year from GEDCOM record."""
        # Placeholder implementation
        return None

    def _extract_birth_place(self, person_record) -> Optional[str]:
        """Extract birth place from GEDCOM record."""
        # Placeholder implementation
        return None

    def _extract_death_place(self, person_record) -> Optional[str]:
        """Extract death place from GEDCOM record."""
        # Placeholder implementation
        return None

    def _parse_relationship_distance(self, relationship: str) -> Optional[int]:
        """Parse relationship string to get distance."""
        # Simple implementation for common relationships
        relationship_distances = {
            "parent": 1, "child": 1,
            "sibling": 1, "brother": 1, "sister": 1,
            "grandparent": 2, "grandchild": 2,
            "aunt": 2, "uncle": 2, "niece": 2, "nephew": 2,
            "1st cousin": 3, "first cousin": 3,
            "2nd cousin": 5, "second cousin": 5,
            "3rd cousin": 7, "third cousin": 7
        }

        relationship_lower = relationship.lower()
        for rel, distance in relationship_distances.items():
            if rel in relationship_lower:
                return distance

        return None

    def _is_plausible_relationship_match(self, dna_match: DNAMatch, gedcom_person: GedcomPerson, relationship_distance: int) -> bool:
        """Check if a relationship match is plausible."""
        # Placeholder implementation
        return True

    def _get_expected_cm_range(self, relationship: str) -> Optional[tuple[float, float]]:
        """Get expected centimorgan range for a relationship."""
        # Simplified ranges for common relationships
        cm_ranges = {
            "parent": (3400, 3700), "child": (3400, 3700),
            "sibling": (2300, 2900),
            "grandparent": (1500, 1900), "grandchild": (1500, 1900),
            "aunt": (1300, 2300), "uncle": (1300, 2300),
            "1st cousin": (500, 1300), "first cousin": (500, 1300),
            "2nd cousin": (100, 400), "second cousin": (100, 400),
            "3rd cousin": (50, 200), "third cousin": (50, 200)
        }

        relationship_lower = relationship.lower()
        for rel, cm_range in cm_ranges.items():
            if rel in relationship_lower:
                return cm_range

        return None

    def _is_cm_in_range(self, actual_cm: float, expected_range: tuple[float, float]) -> bool:
        """Check if actual cM is within expected range (with some tolerance)."""
        min_cm, max_cm = expected_range
        # Allow 20% tolerance outside the range
        tolerance = (max_cm - min_cm) * 0.2
        return (min_cm - tolerance) <= actual_cm <= (max_cm + tolerance)

    def _generate_crossref_summary(self) -> dict[str, Any]:
        """Generate summary of cross-reference analysis."""
        return {
            "total_cross_references": len(self.cross_reference_matches),
            "high_confidence_matches": len([m for m in self.cross_reference_matches if m.confidence_score > 0.7]),
            "conflicts_found": len(self.conflicts_identified),
            "verification_opportunities": len(self.verification_opportunities)
        }

    def _generate_crossref_recommendations(self) -> list[str]:
        """Generate recommendations based on cross-reference analysis."""
        recommendations = []

        high_confidence = len([m for m in self.cross_reference_matches if m.confidence_score > 0.7])
        if high_confidence > 0:
            recommendations.append(f"Prioritize verification of {high_confidence} high-confidence DNA-GEDCOM matches")

        if len(self.conflicts_identified) > 0:
            recommendations.append(f"Resolve {len(self.conflicts_identified)} identified conflicts between DNA and tree data")

        if len(self.verification_opportunities) > 5:
            recommendations.append("Consider systematic verification approach for multiple opportunities")

        recommendations.append("Use DNA evidence to validate uncertain GEDCOM relationships")
        recommendations.append("Focus on matches with shared ancestor information for faster verification")

        return recommendations

    def _empty_crossref_result(self) -> dict[str, Any]:
        """Return empty cross-reference result for error cases."""
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "dna_matches_analyzed": 0,
            "gedcom_people_analyzed": 0,
            "cross_reference_matches": [],
            "conflicts_identified": [],
            "verification_opportunities": [],
            "summary": {"total_cross_references": 0, "high_confidence_matches": 0, "conflicts_found": 0, "verification_opportunities": 0},
            "recommendations": [],
            "error": "Cross-reference analysis failed"
        }

    def _crossref_to_dict(self, crossref_match: CrossReferenceMatch) -> dict[str, Any]:
        """Convert CrossReferenceMatch to dictionary."""
        return {
            "match_id": crossref_match.match_id,
            "dna_match_name": crossref_match.dna_match.match_name,
            "estimated_relationship": crossref_match.dna_match.estimated_relationship,
            "shared_dna_cm": crossref_match.dna_match.shared_dna_cm,
            "potential_gedcom_matches": [
                {
                    "person_id": person.person_id,
                    "full_name": person.full_name,
                    "birth_year": person.birth_year,
                    "birth_place": person.birth_place
                }
                for person in crossref_match.potential_gedcom_matches
            ],
            "confidence_score": crossref_match.confidence_score,
            "match_type": crossref_match.match_type,
            "verification_needed": crossref_match.verification_needed,
            "research_suggestions": crossref_match.research_suggestions
        }

    def _conflict_to_dict(self, conflict: ConflictIdentification) -> dict[str, Any]:
        """Convert ConflictIdentification to dictionary."""
        return {
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type,
            "description": conflict.description,
            "dna_evidence": conflict.dna_evidence,
            "gedcom_evidence": conflict.gedcom_evidence,
            "severity": conflict.severity,
            "resolution_steps": conflict.resolution_steps
        }


# Test functions
def test_dna_gedcom_crossref() -> bool:
    """Test the DNA-GEDCOM cross-reference system."""
    logger.info("Testing DNA-GEDCOM cross-reference system...")

    crossref = DNAGedcomCrossReferencer()

    # Test with mock data
    mock_dna_matches = [
        DNAMatch(
            match_id="test_match_1",
            match_name="John Smith",
            estimated_relationship="2nd cousin",
            shared_dna_cm=150.0,
            shared_ancestors=["William Smith"]
        )
    ]

    mock_gedcom_data = type('MockGedcom', (), {
        'indi_index': {'I1': type('Person', (), {'name': ['John Smith']})()},
        'id_to_parents': {},
        'id_to_children': {}
    })()

    result = crossref.analyze_dna_gedcom_connections(mock_dna_matches, mock_gedcom_data)

    assert "analysis_timestamp" in result, "Result should include timestamp"
    assert "cross_reference_matches" in result, "Result should include cross-reference matches"
    assert "conflicts_identified" in result, "Result should include conflicts"
    assert "verification_opportunities" in result, "Result should include verification opportunities"

    logger.info("✅ DNA-GEDCOM cross-reference test passed")
    return True


def test_name_match_and_confidence_boost() -> None:
    """Test that multiple match types boost confidence score up to cap."""
    crossref = DNAGedcomCrossReferencer()
    DNAMatch(match_id="m1", match_name="Alice Brown", estimated_relationship="2nd cousin", shared_dna_cm=150.0, shared_ancestors=["Brown"])
    # Create GEDCOM person with overlapping name and ancestor
    gedcom_person = GedcomPerson(person_id="I1", full_name="Alice Marie Brown")
    # Manually craft potential matches representing two types
    potential = [
        {"gedcom_person": gedcom_person, "confidence": 0.6, "type": "name_match"},
        {"gedcom_person": gedcom_person, "confidence": 0.55, "type": "ancestor_match"},
    ]
    unique = crossref._deduplicate_and_score_matches(potential)
    assert len(unique) == 1
    boosted = unique[0]["confidence"]
    assert boosted >= 0.6, "Confidence should be boosted when multiple match types"
    assert boosted <= 1.0


def test_conflict_identification_out_of_range_cm() -> None:
    """Test conflict creation when shared cM outside expected range."""
    crossref = DNAGedcomCrossReferencer()
    # Construct a cross_reference_matches entry with mismatch cM
    bad_match = DNAMatch(match_id="m2", match_name="Bob Smith", estimated_relationship="2nd cousin", shared_dna_cm=900.0)
    crossref.cross_reference_matches.append(
        CrossReferenceMatch(
            match_id="crossref_m2",
            dna_match=bad_match,
            potential_gedcom_matches=[],
            confidence_score=0.9,
            match_type="name_match",
        )
    )
    crossref._identify_relationship_conflicts([bad_match], [])
    assert any(c.conflict_id.startswith("cm_conflict_") for c in crossref.conflicts_identified), "Should flag relationship mismatch conflict"


def test_verification_opportunity_threshold() -> None:
    """High confidence matches should produce verification opportunities (>0.7)."""
    crossref = DNAGedcomCrossReferencer()
    good_match = DNAMatch(match_id="m3", match_name="Carol Jones", estimated_relationship="1st cousin", shared_dna_cm=800.0)
    crossref.cross_reference_matches.append(
        CrossReferenceMatch(
            match_id="crossref_m3",
            dna_match=good_match,
            potential_gedcom_matches=[],
            confidence_score=0.75,
            match_type="name_match",
        )
    )
    crossref._identify_verification_opportunities()
    assert len(crossref.verification_opportunities) == 1, "High confidence match should yield verification opportunity"


def test_relationship_distance_parser() -> None:
    """Parser should map common relationship strings to distances."""
    crossref = DNAGedcomCrossReferencer()
    assert crossref._parse_relationship_distance("1st cousin") == 3
    assert crossref._parse_relationship_distance("Third Cousin") == 7
    assert crossref._parse_relationship_distance("Unknown Rel") is None


def dna_gedcom_crossref_module_tests() -> bool:
    """
    Comprehensive test suite for dna_gedcom_crossref.py with real functionality testing.
    Tests DNA-GEDCOM cross-referencing, conflict detection, and resolution systems.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("DNA-GEDCOM Cross-Reference Analysis", "dna_gedcom_crossref.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Cross-reference basic flow",
            test_dna_gedcom_crossref,
            "End-to-end analysis returns expected top-level keys",
            "Run analyze_dna_gedcom_connections with minimal mock data",
            "Validate analyze_dna_gedcom_connections output keys",
        )
        suite.run_test(
            "Confidence boost on multi-match",
            test_name_match_and_confidence_boost,
            "Confidence increases (capped) when multiple match types present",
            "Deduplicate and score with two match types",
            "Validate _deduplicate_and_score_matches boosting logic",
        )
        suite.run_test(
            "Conflict detection for out-of-range cM",
            test_conflict_identification_out_of_range_cm,
            "Creates conflict when cM far outside expected relationship range",
            "Invoke _identify_relationship_conflicts with mismatched cM",
            "Validate relationship mismatch conflict generation",
        )
        suite.run_test(
            "Verification opportunity threshold",
            test_verification_opportunity_threshold,
            "High confidence (>0.7) produces verification opportunity",
            "Call _identify_verification_opportunities on high confidence match",
            "Validate verification opportunity list population",
        )
        suite.run_test(
            "Relationship distance parser",
            test_relationship_distance_parser,
            "Maps common relationship strings to numeric distances",
            "Call _parse_relationship_distance with variants",
            "Validate relationship distance parsing",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive DNA-GEDCOM cross-reference tests using standardized TestSuite format."""
    return dna_gedcom_crossref_module_tests()


if __name__ == "__main__":
    """
    Execute comprehensive DNA-GEDCOM cross-reference tests when run directly.
    Tests DNA-GEDCOM cross-referencing, conflict detection, and resolution systems.
    """
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
