"""
Triangulation Intelligence Module.

Advanced analysis of DNA matches to identify triangulation opportunities,
generate relationship hypotheses, and score confidence levels based on
shared DNA segments, tree data, and historical conversation context.

Features:
- Hypothesis scoring based on multiple evidence sources
- Confidence level calculation with weighted factors
- Cluster analysis for match groups
- Integration with existing TriangulationService
"""

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional

from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for triangulation hypotheses."""

    VERY_HIGH = "very_high"  # 90%+ confidence
    HIGH = "high"  # 75-90% confidence
    MODERATE = "moderate"  # 50-75% confidence
    LOW = "low"  # 25-50% confidence
    SPECULATIVE = "speculative"  # <25% confidence


class EvidenceType(Enum):
    """Types of evidence supporting a hypothesis."""

    DNA_SEGMENT = "dna_segment"  # Shared DNA segment data
    TREE_MATCH = "tree_match"  # Match found in family tree
    SURNAME_MATCH = "surname_match"  # Matching surnames
    LOCATION_MATCH = "location_match"  # Matching geographic locations
    TIMEFRAME_MATCH = "timeframe_match"  # Overlapping time periods
    CONVERSATION = "conversation"  # Information from past conversations
    SHARED_MATCH = "shared_match"  # Common matches with other people


@dataclass
class Evidence:
    """Individual piece of evidence supporting a hypothesis."""

    evidence_type: EvidenceType
    description: str
    weight: float  # 0.0 to 1.0
    source: str  # Where this evidence came from
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriangulationHypothesis:
    """A hypothesis about how two DNA matches are related."""

    target_uuid: str
    match_uuid: str
    match_name: str
    proposed_relationship: str
    common_ancestor_name: Optional[str]
    common_ancestor_id: Optional[str]
    evidence: list[Evidence]
    confidence_score: float  # 0.0 to 100.0
    confidence_level: ConfidenceLevel
    suggested_message: str
    notes: list[str] = field(default_factory=list)


@dataclass
class ClusterAnchor:
    """
    A match with confirmed tree placement that anchors a cluster.

    Phase 5.3: Anchors are high-confidence reference points that help
    identify the common ancestor for a cluster of matches.
    """

    match_uuid: str
    match_name: str
    shared_cm: float
    confirmed_ancestor_name: Optional[str] = None
    tree_depth: int = 0  # How many generations back the connection is known
    has_linked_tree: bool = False
    relationship_path: Optional[str] = None  # e.g., "3rd cousin via John Smith"


@dataclass
class ClusterResearchHypothesis:
    """
    Research hypothesis for an entire cluster.

    Phase 5.3: Suggests research directions based on cluster characteristics.
    """

    hypothesis_id: str
    description: str
    suggested_actions: list[str]
    priority: str  # "high", "medium", "low"
    target_ancestor: Optional[str] = None
    evidence_summary: str = ""


@dataclass
class MatchCluster:
    """A group of matches that likely share a common ancestor."""

    cluster_id: str
    common_ancestor_hypothesis: Optional[str]
    match_uuids: list[str]
    total_shared_cm: float
    average_shared_cm: float
    confidence_level: ConfidenceLevel
    evidence: list[Evidence]
    # Phase 5.3 additions
    anchors: list[ClusterAnchor] = field(default_factory=list)
    research_hypotheses: list[ClusterResearchHypothesis] = field(default_factory=list)


class TriangulationIntelligence:
    """
    Advanced triangulation analysis engine.

    Analyzes DNA matches to identify triangulation opportunities,
    generate hypotheses, and calculate confidence scores.
    """

    # Confidence thresholds
    VERY_HIGH_THRESHOLD = 90.0
    HIGH_THRESHOLD = 75.0
    MODERATE_THRESHOLD = 50.0
    LOW_THRESHOLD = 25.0

    # Evidence weights
    EVIDENCE_WEIGHTS: ClassVar[dict[EvidenceType, float]] = {
        EvidenceType.DNA_SEGMENT: 0.35,
        EvidenceType.TREE_MATCH: 0.25,
        EvidenceType.SHARED_MATCH: 0.15,
        EvidenceType.SURNAME_MATCH: 0.10,
        EvidenceType.LOCATION_MATCH: 0.08,
        EvidenceType.TIMEFRAME_MATCH: 0.05,
        EvidenceType.CONVERSATION: 0.02,
    }

    def __init__(
        self,
        db_session: Optional[Any] = None,
        research_service: Optional[Any] = None,
    ):
        """
        Initialize the triangulation intelligence engine.

        Args:
            db_session: SQLAlchemy database session
            research_service: ResearchService for GEDCOM/tree queries
        """
        self.db_session = db_session
        self.research_service = research_service

    def analyze_match(
        self,
        target_uuid: str,
        match_uuid: str,
        match_data: dict[str, Any],
    ) -> TriangulationHypothesis:
        """
        Analyze a single DNA match and generate a hypothesis.

        Args:
            target_uuid: UUID of the target person
            match_uuid: UUID of the DNA match
            match_data: Dictionary with match information (name, shared_cm, etc.)

        Returns:
            TriangulationHypothesis with confidence score
        """
        evidence: list[Evidence] = []
        notes: list[str] = []

        # Collect evidence from various sources
        evidence.extend(self._analyze_dna_evidence(match_data))
        evidence.extend(self._analyze_tree_evidence(match_uuid, match_data))
        evidence.extend(self._analyze_surname_evidence(match_data))
        evidence.extend(self._analyze_location_evidence(match_data))
        evidence.extend(self._analyze_shared_matches(target_uuid, match_uuid))

        # Calculate confidence score
        confidence_score = self._calculate_confidence(evidence)
        confidence_level = self._get_confidence_level(confidence_score)

        # Determine common ancestor
        common_ancestor = self._identify_common_ancestor(evidence)

        # Generate hypothesis details
        proposed_relationship = self._propose_relationship(match_data)
        suggested_message = self._generate_message(
            match_data.get("name", "Unknown"),
            proposed_relationship,
            common_ancestor,
        )

        return TriangulationHypothesis(
            target_uuid=target_uuid,
            match_uuid=match_uuid,
            match_name=match_data.get("name", "Unknown"),
            proposed_relationship=proposed_relationship,
            common_ancestor_name=common_ancestor.get("name") if common_ancestor else None,
            common_ancestor_id=common_ancestor.get("id") if common_ancestor else None,
            evidence=evidence,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            suggested_message=suggested_message,
            notes=notes,
        )

    def find_clusters(
        self,
        matches: list[dict[str, Any]],
        min_cluster_size: int = 3,
    ) -> list[MatchCluster]:
        """
        Identify clusters of matches that likely share a common ancestor.

        Args:
            matches: List of match dictionaries
            min_cluster_size: Minimum matches to form a cluster

        Returns:
            List of MatchCluster objects
        """
        clusters: list[MatchCluster] = []

        # Group by surname patterns
        surname_groups = self._group_by_surnames(matches)

        for surname, group_matches in surname_groups.items():
            if len(group_matches) >= min_cluster_size:
                cluster = self._create_cluster(surname, group_matches)
                clusters.append(cluster)

        return clusters

    def prioritize_hypotheses(
        self,
        hypotheses: list[TriangulationHypothesis],
        max_results: int = 10,
    ) -> list[TriangulationHypothesis]:
        """
        Prioritize hypotheses by confidence and actionability.

        Args:
            hypotheses: List of hypotheses to prioritize
            max_results: Maximum number to return

        Returns:
            Sorted list of top hypotheses
        """
        # Score each hypothesis for actionability
        scored: list[tuple[TriangulationHypothesis, float]] = []
        for h in hypotheses:
            action_score = self._calculate_actionability(h)
            scored.append((h, action_score))

        # Sort by combined score (confidence * actionability)
        scored.sort(key=lambda x: x[0].confidence_score * x[1], reverse=True)

        return [h for h, _ in scored[:max_results]]

    @staticmethod
    def _analyze_dna_evidence(match_data: dict[str, Any]) -> list[Evidence]:
        """Analyze DNA-based evidence."""
        evidence: list[Evidence] = []
        shared_cm = match_data.get("shared_cm", 0)

        if shared_cm > 0:
            # Higher shared cM = stronger evidence
            if shared_cm >= 400:
                weight = 1.0
                desc = f"High DNA sharing ({shared_cm} cM) - likely close relative"
            elif shared_cm >= 200:
                weight = 0.85
                desc = f"Moderate DNA sharing ({shared_cm} cM) - likely 2nd-3rd cousin"
            elif shared_cm >= 90:
                weight = 0.70
                desc = f"Moderate DNA sharing ({shared_cm} cM) - likely 3rd-4th cousin"
            elif shared_cm >= 40:
                weight = 0.50
                desc = f"Lower DNA sharing ({shared_cm} cM) - likely 4th-5th cousin"
            else:
                weight = 0.30
                desc = f"Low DNA sharing ({shared_cm} cM) - distant relationship"

            evidence.append(
                Evidence(
                    evidence_type=EvidenceType.DNA_SEGMENT,
                    description=desc,
                    weight=weight,
                    source="DNA Match Data",
                    details={"shared_cm": shared_cm},
                )
            )

        return evidence

    def _analyze_tree_evidence(
        self,
        match_uuid: str,
        match_data: dict[str, Any],
    ) -> list[Evidence]:
        """Analyze evidence from family tree data."""
        evidence: list[Evidence] = []

        # Check if match has a linked tree
        tree_size = match_data.get("tree_size", 0)
        has_tree = match_data.get("has_tree", False) or tree_size > 0

        if has_tree and tree_size > 0:
            if tree_size >= 500:
                weight = 0.90
                desc = f"Large tree ({tree_size} people) - high research potential"
            elif tree_size >= 100:
                weight = 0.70
                desc = f"Medium tree ({tree_size} people) - good research potential"
            else:
                weight = 0.40
                desc = f"Small tree ({tree_size} people) - limited research potential"

            evidence.append(
                Evidence(
                    evidence_type=EvidenceType.TREE_MATCH,
                    description=desc,
                    weight=weight,
                    source="Ancestry Tree Data",
                    details={"tree_size": tree_size},
                )
            )

        # Check for tree connection via research service
        if self.research_service and match_uuid:
            try:
                path = self.research_service.get_relationship_path("ROOT", match_uuid)
                if path:
                    evidence.append(
                        Evidence(
                            evidence_type=EvidenceType.TREE_MATCH,
                            description=f"Found in family tree with {len(path)} generation path",
                            weight=0.95,
                            source="GEDCOM Tree",
                            details={"path_length": len(path)},
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not check tree path: {e}")

        return evidence

    @staticmethod
    def _analyze_surname_evidence(match_data: dict[str, Any]) -> list[Evidence]:
        """Analyze surname-based evidence."""
        evidence: list[Evidence] = []

        match_surnames = match_data.get("surnames", [])
        target_surnames = match_data.get("target_surnames", [])

        if match_surnames and target_surnames:
            # Check for surname overlaps
            common = {s.lower() for s in match_surnames} & {s.lower() for s in target_surnames}
            if common:
                weight = min(0.90, 0.30 * len(common))
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.SURNAME_MATCH,
                        description=f"Matching surnames: {', '.join(common)}",
                        weight=weight,
                        source="Surname Analysis",
                        details={"common_surnames": list(common)},
                    )
                )

        return evidence

    @staticmethod
    def _analyze_location_evidence(match_data: dict[str, Any]) -> list[Evidence]:
        """Analyze geographic location evidence."""
        evidence: list[Evidence] = []

        match_locations = match_data.get("locations", [])
        target_locations = match_data.get("target_locations", [])

        if match_locations and target_locations:
            # Simple string matching for locations
            common = {loc.lower() for loc in match_locations} & {loc.lower() for loc in target_locations}
            if common:
                weight = min(0.80, 0.25 * len(common))
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.LOCATION_MATCH,
                        description=f"Matching locations: {', '.join(common)}",
                        weight=weight,
                        source="Location Analysis",
                        details={"common_locations": list(common)},
                    )
                )

        return evidence

    def _analyze_shared_matches(
        self,
        target_uuid: str,
        match_uuid: str,
    ) -> list[Evidence]:
        """Analyze shared matches between target and match."""
        evidence: list[Evidence] = []

        if not self.db_session:
            return evidence

        try:
            from sqlalchemy import func

            from core.database import Person, SharedMatch

            # Get person IDs from UUIDs
            target_person = self.db_session.query(Person.id).filter(Person.uuid == target_uuid.upper()).scalar()
            match_person = self.db_session.query(Person.id).filter(Person.uuid == match_uuid.upper()).scalar()

            if not target_person or not match_person:
                return evidence

            # Query shared matches count between target and match
            shared_match_count = (
                self.db_session.query(func.count(SharedMatch.id))
                .filter(
                    SharedMatch.person_id == match_person,
                )
                .scalar()
                or 0
            )

            if shared_match_count > 0:
                if shared_match_count >= 10:
                    weight = 0.85
                    desc = f"High shared match count ({shared_match_count}) - strong triangulation"
                elif shared_match_count >= 5:
                    weight = 0.65
                    desc = f"Moderate shared matches ({shared_match_count}) - good triangulation"
                else:
                    weight = 0.40
                    desc = f"Some shared matches ({shared_match_count}) - possible triangulation"

                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.SHARED_MATCH,
                        description=desc,
                        weight=weight,
                        source="Shared Match Analysis",
                        details={
                            "shared_match_count": shared_match_count,
                            "target_uuid": target_uuid,
                            "match_uuid": match_uuid,
                        },
                    )
                )
        except Exception as e:
            logger.debug(f"Could not analyze shared matches: {e}")

        return evidence

    def _calculate_confidence(self, evidence: list[Evidence]) -> float:
        """Calculate overall confidence score from evidence."""
        if not evidence:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for e in evidence:
            type_weight = self.EVIDENCE_WEIGHTS.get(e.evidence_type, 0.05)
            contribution = e.weight * type_weight * 100
            weighted_sum += contribution
            total_weight += type_weight

        if total_weight == 0:
            return 0.0

        # Normalize to 0-100 scale
        return min(100.0, weighted_sum / total_weight)

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= self.VERY_HIGH_THRESHOLD:
            return ConfidenceLevel.VERY_HIGH
        if score >= self.HIGH_THRESHOLD:
            return ConfidenceLevel.HIGH
        if score >= self.MODERATE_THRESHOLD:
            return ConfidenceLevel.MODERATE
        if score >= self.LOW_THRESHOLD:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.SPECULATIVE

    @staticmethod
    def _identify_common_ancestor(
        evidence: list[Evidence],
    ) -> Optional[dict[str, Any]]:
        """Identify the most likely common ancestor from evidence."""
        # Check for explicit tree match evidence
        for e in evidence:
            if e.evidence_type == EvidenceType.TREE_MATCH and "ancestor" in e.details:
                return e.details["ancestor"]

        # Check for surname-based hypothesis
        for e in evidence:
            if e.evidence_type == EvidenceType.SURNAME_MATCH:
                surnames = e.details.get("common_surnames", [])
                if surnames:
                    return {"name": f"{surnames[0].title()} ancestor", "source": "surname"}

        return None

    @staticmethod
    def _propose_relationship(match_data: dict[str, Any]) -> str:
        """Propose a relationship type based on shared DNA."""
        shared_cm = match_data.get("shared_cm", 0)

        # Standard relationship estimates based on shared cM thresholds
        thresholds = [
            (1450, "Close family (parent/child/sibling)"),
            (680, "1st cousin or closer"),
            (200, "2nd cousin"),
            (90, "3rd cousin"),
            (40, "4th cousin"),
            (20, "5th cousin"),
        ]
        for threshold, relationship in thresholds:
            if shared_cm >= threshold:
                return relationship
        return "Distant cousin"

    @staticmethod
    def _generate_message(
        match_name: str,
        relationship: str,
        common_ancestor: Optional[dict[str, Any]],
    ) -> str:
        """Generate a suggested outreach message."""
        if common_ancestor:
            ancestor_name = common_ancestor.get("name", "our common ancestor")
            return (
                f"Hi {match_name}! Based on our DNA match and family tree research, "
                f"I believe we may be {relationship}s, possibly connected through {ancestor_name}. "
                f"I'd love to compare notes on our family histories!"
            )
        return (
            f"Hi {match_name}! Our DNA match suggests we may be {relationship}s. "
            f"I'm researching our potential connection and would love to compare family trees!"
        )

    @staticmethod
    def _group_by_surnames(
        matches: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group matches by their surnames."""
        groups: dict[str, list[dict[str, Any]]] = {}

        for match in matches:
            surnames = match.get("surnames", [])
            for surname in surnames:
                key = surname.lower()
                if key not in groups:
                    groups[key] = []
                groups[key].append(match)

        return groups

    @staticmethod
    def _create_cluster(
        surname: str,
        matches: list[dict[str, Any]],
    ) -> MatchCluster:
        """Create a match cluster from grouped matches."""
        total_cm = sum(m.get("shared_cm", 0) for m in matches)
        avg_cm = total_cm / len(matches) if matches else 0

        # Determine confidence based on cluster characteristics
        if len(matches) >= 5 and avg_cm >= 50:
            confidence = ConfidenceLevel.HIGH
        elif len(matches) >= 3 and avg_cm >= 30:
            confidence = ConfidenceLevel.MODERATE
        else:
            confidence = ConfidenceLevel.LOW

        evidence = [
            Evidence(
                evidence_type=EvidenceType.SURNAME_MATCH,
                description=f"Cluster of {len(matches)} matches with surname '{surname}'",
                weight=0.7,
                source="Cluster Analysis",
                details={"surname": surname, "match_count": len(matches)},
            )
        ]

        # Phase 5.3: Identify cluster anchors
        anchors = TriangulationIntelligence._identify_cluster_anchors(matches)

        # Phase 5.3: Generate research hypotheses
        research_hypotheses = TriangulationIntelligence._generate_cluster_hypotheses(
            surname, matches, anchors, confidence
        )

        return MatchCluster(
            cluster_id=f"cluster_{surname}_{len(matches)}",
            common_ancestor_hypothesis=f"Ancestor with surname {surname.title()}",
            match_uuids=[m.get("uuid", "") for m in matches],
            total_shared_cm=total_cm,
            average_shared_cm=avg_cm,
            confidence_level=confidence,
            evidence=evidence,
            anchors=anchors,
            research_hypotheses=research_hypotheses,
        )

    @staticmethod
    def _identify_cluster_anchors(matches: list[dict[str, Any]]) -> list[ClusterAnchor]:
        """
        Identify anchor matches with confirmed tree placement.

        Phase 5.3: Anchors are matches with linked trees or confirmed
        ancestor information that help place the cluster in the family tree.

        Args:
            matches: List of match dictionaries

        Returns:
            List of ClusterAnchor objects sorted by confidence
        """
        anchors: list[ClusterAnchor] = []

        for match in matches:
            # Check for tree linkage indicators
            has_tree = match.get("has_linked_tree", False) or match.get("tree_size", 0) > 0
            confirmed_ancestor = match.get("common_ancestor_name") or match.get("mrca")
            shared_cm = match.get("shared_cm", 0)
            relationship = match.get("relationship_path") or match.get("relationship")

            # A match is an anchor if it has a linked tree or confirmed ancestor
            if has_tree or confirmed_ancestor:
                anchor = ClusterAnchor(
                    match_uuid=match.get("uuid", ""),
                    match_name=match.get("name", "Unknown"),
                    shared_cm=shared_cm,
                    confirmed_ancestor_name=confirmed_ancestor,
                    tree_depth=match.get("tree_depth", 0),
                    has_linked_tree=has_tree,
                    relationship_path=relationship,
                )
                anchors.append(anchor)

        # Sort anchors by shared_cm (higher = more reliable)
        anchors.sort(key=lambda a: a.shared_cm, reverse=True)
        return anchors

    @staticmethod
    def _generate_cluster_hypotheses(
        surname: str,
        matches: list[dict[str, Any]],
        anchors: list[ClusterAnchor],
        confidence: ConfidenceLevel,
    ) -> list[ClusterResearchHypothesis]:
        """
        Generate research hypotheses for a cluster.

        Phase 5.3: Creates actionable research suggestions based on
        cluster characteristics and anchor information.

        Args:
            surname: Common surname in cluster
            matches: List of match dictionaries
            anchors: List of identified anchors
            confidence: Cluster confidence level

        Returns:
            List of ClusterResearchHypothesis objects
        """
        hypotheses: list[ClusterResearchHypothesis] = []
        match_count = len(matches)
        total_cm = sum(m.get("shared_cm", 0) for m in matches)

        # Hypothesis 1: Based on anchors
        if anchors:
            best_anchor = anchors[0]
            if best_anchor.confirmed_ancestor_name:
                hypotheses.append(
                    ClusterResearchHypothesis(
                        hypothesis_id=f"h_{surname}_anchor",
                        description=f"Cluster shares ancestor {best_anchor.confirmed_ancestor_name}",
                        suggested_actions=[
                            f"Review {best_anchor.match_name}'s tree for {best_anchor.confirmed_ancestor_name}",
                            f"Search for descendants of {best_anchor.confirmed_ancestor_name} in other matches",
                            "Compare shared segments between cluster members",
                        ],
                        priority="high",
                        target_ancestor=best_anchor.confirmed_ancestor_name,
                        evidence_summary=f"Anchor match ({best_anchor.shared_cm:.0f} cM) confirms connection",
                    )
                )

        # Hypothesis 2: Based on surname pattern
        if match_count >= 3:
            priority = "high" if confidence == ConfidenceLevel.HIGH else "medium"
            hypotheses.append(
                ClusterResearchHypothesis(
                    hypothesis_id=f"h_{surname}_pattern",
                    description=f"{match_count} matches share {surname.title()} surname pattern",
                    suggested_actions=[
                        f"Research {surname.title()} family lines in your tree",
                        f"Check for {surname.title()} in vital records (births, marriages, deaths)",
                        "Look for geographic clustering among these matches",
                    ],
                    priority=priority,
                    target_ancestor=f"Unknown {surname.title()} ancestor",
                    evidence_summary=f"{match_count} matches, {total_cm:.0f} total cM shared",
                )
            )

        # Hypothesis 3: Geographic research (if locations available)
        locations = [m.get("location", "") for m in matches if m.get("location")]
        if locations:
            unique_locations = list({loc for loc in locations if loc})[:3]
            if unique_locations:
                hypotheses.append(
                    ClusterResearchHypothesis(
                        hypothesis_id=f"h_{surname}_geo",
                        description=f"Geographic focus: {', '.join(unique_locations)}",
                        suggested_actions=[
                            f"Research {surname.title()} records in {unique_locations[0]}",
                            "Check immigration/emigration records for this surname",
                            "Look for church records in these locations",
                        ],
                        priority="medium",
                        evidence_summary=f"Matches concentrated in {len(unique_locations)} location(s)",
                    )
                )

        return hypotheses

    @staticmethod
    def _calculate_actionability(hypothesis: TriangulationHypothesis) -> float:
        """Calculate how actionable a hypothesis is."""
        score = 0.5  # Base score

        # Boost for having a common ancestor identified
        if hypothesis.common_ancestor_name:
            score += 0.2

        # Boost for higher confidence
        if hypothesis.confidence_level == ConfidenceLevel.VERY_HIGH:
            score += 0.3
        elif hypothesis.confidence_level == ConfidenceLevel.HIGH:
            score += 0.2
        elif hypothesis.confidence_level == ConfidenceLevel.MODERATE:
            score += 0.1

        return min(1.0, score)


# --- Test Suite ---


def _test_confidence_level_calculation() -> None:
    """Test confidence level boundaries."""
    engine = TriangulationIntelligence()

    assert engine._get_confidence_level(95.0) == ConfidenceLevel.VERY_HIGH
    assert engine._get_confidence_level(80.0) == ConfidenceLevel.HIGH
    assert engine._get_confidence_level(60.0) == ConfidenceLevel.MODERATE
    assert engine._get_confidence_level(30.0) == ConfidenceLevel.LOW
    assert engine._get_confidence_level(10.0) == ConfidenceLevel.SPECULATIVE


def _test_dna_evidence_analysis() -> None:
    """Test DNA evidence weight calculation."""
    engine = TriangulationIntelligence()

    # High shared cM
    evidence = engine._analyze_dna_evidence({"shared_cm": 500})
    assert len(evidence) == 1
    assert evidence[0].weight == 1.0
    assert evidence[0].evidence_type == EvidenceType.DNA_SEGMENT

    # Low shared cM
    evidence = engine._analyze_dna_evidence({"shared_cm": 20})
    assert len(evidence) == 1
    assert evidence[0].weight == 0.30

    # No shared cM
    evidence = engine._analyze_dna_evidence({})
    assert len(evidence) == 0


def _test_hypothesis_generation() -> None:
    """Test hypothesis generation with match data."""
    engine = TriangulationIntelligence()

    match_data = {
        "name": "John Smith",
        "shared_cm": 150,
        "tree_size": 200,
        "has_tree": True,
    }

    hypothesis = engine.analyze_match("target-uuid", "match-uuid", match_data)

    assert hypothesis.match_name == "John Smith"
    assert hypothesis.confidence_score > 0
    assert hypothesis.proposed_relationship, "Should have proposed relationship"
    assert hypothesis.suggested_message, "Should have suggested message"


def _test_cluster_detection() -> None:
    """Test match clustering by surname."""
    engine = TriangulationIntelligence()

    matches = [
        {"uuid": "1", "surnames": ["Smith", "Jones"], "shared_cm": 50},
        {"uuid": "2", "surnames": ["Smith"], "shared_cm": 60},
        {"uuid": "3", "surnames": ["Smith", "Brown"], "shared_cm": 40},
        {"uuid": "4", "surnames": ["Johnson"], "shared_cm": 30},
    ]

    clusters = engine.find_clusters(matches, min_cluster_size=3)

    assert len(clusters) == 1
    assert clusters[0].common_ancestor_hypothesis is not None
    assert len(clusters[0].match_uuids) == 3


def _test_relationship_proposal() -> None:
    """Test relationship type based on shared cM."""
    engine = TriangulationIntelligence()

    assert "1st cousin" in engine._propose_relationship({"shared_cm": 800})
    assert "2nd cousin" in engine._propose_relationship({"shared_cm": 250})
    assert "3rd cousin" in engine._propose_relationship({"shared_cm": 100})
    assert "4th cousin" in engine._propose_relationship({"shared_cm": 50})
    assert "Distant" in engine._propose_relationship({"shared_cm": 10})


def _test_prioritization() -> None:
    """Test hypothesis prioritization."""
    engine = TriangulationIntelligence()

    h1 = TriangulationHypothesis(
        target_uuid="t1",
        match_uuid="m1",
        match_name="Match 1",
        proposed_relationship="2nd cousin",
        common_ancestor_name="John Smith",
        common_ancestor_id="ca1",
        evidence=[],
        confidence_score=85.0,
        confidence_level=ConfidenceLevel.HIGH,
        suggested_message="Hi!",
    )

    h2 = TriangulationHypothesis(
        target_uuid="t1",
        match_uuid="m2",
        match_name="Match 2",
        proposed_relationship="4th cousin",
        common_ancestor_name=None,
        common_ancestor_id=None,
        evidence=[],
        confidence_score=45.0,
        confidence_level=ConfidenceLevel.LOW,
        suggested_message="Hello!",
    )

    prioritized = engine.prioritize_hypotheses([h2, h1])

    assert len(prioritized) == 2
    assert prioritized[0].match_name == "Match 1"  # Higher confidence first


def _test_cluster_anchor_identification() -> None:
    """Test Phase 5.3 anchor identification in clusters."""
    matches = [
        {"uuid": "m1", "name": "Anchor Match", "shared_cm": 150, "has_linked_tree": True, "common_ancestor_name": "John Smith"},
        {"uuid": "m2", "name": "Regular Match", "shared_cm": 80},
        {"uuid": "m3", "name": "Tree Match", "shared_cm": 60, "tree_size": 500},
    ]

    anchors = TriangulationIntelligence._identify_cluster_anchors(matches)

    assert len(anchors) == 2, "Should identify 2 anchors"
    assert anchors[0].match_name == "Anchor Match", "Highest shared_cm anchor first"
    assert anchors[0].confirmed_ancestor_name == "John Smith"
    assert anchors[0].has_linked_tree is True


def _test_cluster_research_hypotheses() -> None:
    """Test Phase 5.3 hypothesis generation for clusters."""
    matches = [
        {"uuid": "m1", "name": "Match 1", "shared_cm": 100, "has_linked_tree": True, "common_ancestor_name": "Mary Jones"},
        {"uuid": "m2", "name": "Match 2", "shared_cm": 80},
        {"uuid": "m3", "name": "Match 3", "shared_cm": 60},
    ]
    anchors = TriangulationIntelligence._identify_cluster_anchors(matches)
    hypotheses = TriangulationIntelligence._generate_cluster_hypotheses(
        "jones", matches, anchors, ConfidenceLevel.HIGH
    )

    assert len(hypotheses) >= 2, "Should generate at least 2 hypotheses"
    # First hypothesis should be anchor-based
    assert any("Mary Jones" in h.description for h in hypotheses), "Should reference anchor ancestor"
    # Should have suggested actions
    assert all(len(h.suggested_actions) > 0 for h in hypotheses), "Each hypothesis should have actions"


def _test_cluster_with_geographic_hypothesis() -> None:
    """Test Phase 5.3 geographic hypothesis generation."""
    matches = [
        {"uuid": "m1", "name": "Match 1", "shared_cm": 100, "location": "Scotland"},
        {"uuid": "m2", "name": "Match 2", "shared_cm": 80, "location": "Scotland"},
        {"uuid": "m3", "name": "Match 3", "shared_cm": 60, "location": "Ireland"},
    ]
    anchors: list[ClusterAnchor] = []
    hypotheses = TriangulationIntelligence._generate_cluster_hypotheses(
        "smith", matches, anchors, ConfidenceLevel.MODERATE
    )

    # Should have geographic hypothesis
    geo_hypothesis = [h for h in hypotheses if "geo" in h.hypothesis_id]
    assert len(geo_hypothesis) == 1, "Should generate geographic hypothesis"
    assert "Scotland" in geo_hypothesis[0].description or "Ireland" in geo_hypothesis[0].description


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Triangulation Intelligence", "triangulation_intelligence.py")
    suite.start_suite()

    suite.run_test("Confidence level calculation", _test_confidence_level_calculation)
    suite.run_test("DNA evidence weight analysis", _test_dna_evidence_analysis)
    suite.run_test("Hypothesis generation", _test_hypothesis_generation)
    suite.run_test("Match cluster detection", _test_cluster_detection)
    suite.run_test("Relationship type proposal", _test_relationship_proposal)
    suite.run_test("Hypothesis prioritization", _test_prioritization)
    # Phase 5.3 tests
    suite.run_test("Cluster anchor identification", _test_cluster_anchor_identification)
    suite.run_test("Cluster research hypotheses", _test_cluster_research_hypotheses)
    suite.run_test("Geographic hypothesis generation", _test_cluster_with_geographic_hypothesis)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
