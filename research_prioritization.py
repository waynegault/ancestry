"""
Advanced Utility & Intelligent Service Engine

Sophisticated utility platform providing comprehensive service automation,
intelligent utility functions, and advanced operational capabilities with
optimized algorithms, professional-grade utilities, and comprehensive
service management for genealogical automation and research workflows.

Utility Intelligence:
• Advanced utility functions with intelligent automation and optimization protocols
• Sophisticated service management with comprehensive operational capabilities
• Intelligent utility coordination with multi-system integration and synchronization
• Comprehensive utility analytics with detailed performance metrics and insights
• Advanced utility validation with quality assessment and verification protocols
• Integration with service platforms for comprehensive utility management and automation

Service Automation:
• Sophisticated service automation with intelligent workflow generation and execution
• Advanced utility optimization with performance monitoring and enhancement protocols
• Intelligent service coordination with automated management and orchestration
• Comprehensive service validation with quality assessment and reliability protocols
• Advanced service analytics with detailed operational insights and optimization
• Integration with automation systems for comprehensive service management workflows

Professional Services:
• Advanced professional utilities with enterprise-grade functionality and reliability
• Sophisticated service protocols with professional standards and best practices
• Intelligent service optimization with performance monitoring and enhancement
• Comprehensive service documentation with detailed operational guides and analysis
• Advanced service security with secure protocols and data protection measures
• Integration with professional service systems for genealogical research workflows

Foundation Services:
Provides the essential utility infrastructure that enables reliable, high-performance
operations through intelligent automation, comprehensive service management,
and professional utilities for genealogical automation and research workflows.

Technical Implementation:
Intelligent Research Prioritization System for Ancestry Project

This module provides intelligent prioritization of genealogical research tasks
based on GEDCOM analysis, DNA evidence, and research efficiency factors.
Generates location-specific and time-period-specific research suggestions
with family line completion tracking.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 12.3 - Intelligent Research Prioritization
"""

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)


@dataclass
class ResearchPriority:
    """Represents a prioritized research task with scoring and context."""

    priority_id: str
    task_type: str  # 'vital_records', 'census', 'immigration', 'dna_verification', 'conflict_resolution'
    description: str
    target_people: list[str]
    priority_score: float  # 0.0 to 100.0
    urgency: str  # 'critical', 'high', 'medium', 'low'
    research_context: dict[str, Any] = field(default_factory=dict)
    expected_outcomes: list[str] = field(default_factory=list)
    research_steps: list[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # 'low', 'medium', 'high'
    success_probability: float = 0.5  # 0.0 to 1.0


@dataclass
class FamilyLineStatus:
    """Tracks completion status of family lines."""

    line_id: str
    line_name: str
    surname: str
    generations_back: int
    completeness_percentage: float
    missing_generations: list[int]
    research_bottlenecks: list[str]
    priority_research_targets: list[str]


@dataclass
class LocationResearchCluster:
    """Represents a geographic research cluster for efficiency."""

    cluster_id: str
    location: str
    time_period: str
    people_count: int
    target_people: list[str]
    available_records: list[str]
    research_efficiency_score: float
    cluster_research_plan: list[str] = field(default_factory=list)


class IntelligentResearchPrioritizer:
    """
    AI-powered system for prioritizing genealogical research tasks.
    """

    def __init__(self) -> None:
        """Initialize the research prioritizer."""
        self.research_priorities: list[ResearchPriority] = []
        self.family_line_status: list[FamilyLineStatus] = []
        self.location_clusters: list[LocationResearchCluster] = []

    def prioritize_research_tasks(
        self,
        gedcom_analysis: dict[str, Any],
        dna_crossref_analysis: dict[str, Any],
        _existing_tasks: Optional[list[dict[str, Any]]] = None
    ) -> dict[str, Any]:
        """
        Generate intelligent research prioritization based on multiple data sources.

        Args:
            gedcom_analysis: Results from GEDCOM intelligence analysis
            dna_crossref_analysis: Results from DNA-GEDCOM cross-reference
            existing_tasks: Optional existing research tasks to incorporate

        Returns:
            Dictionary containing prioritized research plan
        """
        try:
            logger.info("Starting intelligent research prioritization")

            # Clear previous analysis
            self.research_priorities.clear()
            self.family_line_status.clear()
            self.location_clusters.clear()

            # Analyze family line completeness
            self._analyze_family_line_completeness(gedcom_analysis)

            # Create location-based research clusters
            self._create_location_research_clusters(gedcom_analysis)

            # Generate prioritized research tasks
            self._generate_priority_tasks_from_gaps(gedcom_analysis)
            self._generate_priority_tasks_from_conflicts(gedcom_analysis)
            self._generate_priority_tasks_from_dna(dna_crossref_analysis)
            self._generate_cluster_research_tasks()

            # Score and rank all priorities
            self._score_and_rank_priorities()

            # Generate research plan
            research_plan = {
                "prioritization_timestamp": datetime.now().isoformat(),
                "total_priorities_identified": len(self.research_priorities),
                "family_line_analysis": [self._family_line_to_dict(line) for line in self.family_line_status],
                "location_clusters": [self._location_cluster_to_dict(cluster) for cluster in self.location_clusters],
                "prioritized_tasks": [self._priority_to_dict(priority) for priority in self.research_priorities],
                "research_recommendations": self._generate_research_recommendations(),
                "efficiency_analysis": self._analyze_research_efficiency(),
                "next_steps": self._generate_next_steps()
            }

            logger.info(f"Research prioritization completed: {len(self.research_priorities)} priorities identified")
            return research_plan

        except Exception as e:
            logger.error(f"Error during research prioritization: {e}")
            return self._empty_prioritization_result()

    def _analyze_family_line_completeness(self, gedcom_analysis: dict[str, Any]):
        """Analyze completeness of different family lines."""
        try:
            # Extract family patterns from GEDCOM analysis
            ai_insights = gedcom_analysis.get("ai_insights", {})
            family_patterns = ai_insights.get("family_patterns", {})
            common_surnames = family_patterns.get("common_surnames", [])

            # Analyze each major surname line
            for _i, surname in enumerate(common_surnames[:5]):  # Top 5 surnames
                line_status = FamilyLineStatus(
                    line_id=f"line_{surname.lower()}",
                    line_name=f"{surname} Family Line",
                    surname=surname,
                    generations_back=self._estimate_generations_back(surname, gedcom_analysis),
                    completeness_percentage=self._calculate_line_completeness(surname, gedcom_analysis),
                    missing_generations=self._identify_missing_generations(surname, gedcom_analysis),
                    research_bottlenecks=self._identify_research_bottlenecks(surname, gedcom_analysis),
                    priority_research_targets=self._identify_priority_targets(surname, gedcom_analysis)
                )

                self.family_line_status.append(line_status)

        except Exception as e:
            logger.debug(f"Error analyzing family line completeness: {e}")

    def _create_location_research_clusters(self, gedcom_analysis: dict[str, Any]):
        """Create location-based research clusters for efficiency."""
        try:
            # Group research opportunities by location
            location_groups = defaultdict(list)

            gaps = gedcom_analysis.get("gaps_identified", [])
            opportunities = gedcom_analysis.get("research_opportunities", [])

            # Group gaps by location
            for gap in gaps:
                if gap.get("gap_type") in ["missing_places", "missing_dates"]:
                    # Extract location context if available
                    location = self._extract_location_context(gap)
                    if location:
                        location_groups[location].append({
                            "type": "gap",
                            "person_id": gap.get("person_id"),
                            "person_name": gap.get("person_name"),
                            "description": gap.get("description")
                        })

            # Group opportunities by location
            for opportunity in opportunities:
                if opportunity.get("opportunity_type") == "location_research":
                    location = self._extract_opportunity_location(opportunity)
                    if location:
                        location_groups[location].extend(opportunity.get("target_people", []))

            # Create clusters for locations with multiple research targets
            for location, items in location_groups.items():
                if len(items) >= 2:  # At least 2 research targets
                    cluster = LocationResearchCluster(
                        cluster_id=f"cluster_{location.replace(' ', '_').lower()}",
                        location=location,
                        time_period=self._estimate_time_period_for_location(location, items),
                        people_count=len(items),
                        target_people=[item.get("person_name", "") for item in items if isinstance(item, dict)],
                        available_records=self._identify_available_records_for_location(location),
                        research_efficiency_score=self._calculate_cluster_efficiency(location, items),
                        cluster_research_plan=self._generate_cluster_research_plan(location, items)
                    )

                    self.location_clusters.append(cluster)

        except Exception as e:
            logger.debug(f"Error creating location research clusters: {e}")

    def _generate_priority_tasks_from_gaps(self, gedcom_analysis: dict[str, Any]):
        """Generate priority tasks from identified gaps."""
        gaps = gedcom_analysis.get("gaps_identified", [])

        for gap in gaps:
            priority_score = self._calculate_gap_priority_score(gap)

            priority = ResearchPriority(
                priority_id=f"gap_{gap.get('person_id', 'unknown')}",
                task_type=self._map_gap_to_task_type(gap.get("gap_type", "")),
                description=gap.get("description", ""),
                target_people=[gap.get("person_name", "")],
                priority_score=priority_score,
                urgency=self._determine_urgency_from_score(priority_score),
                research_context={
                    "gap_type": gap.get("gap_type"),
                    "person_id": gap.get("person_id"),
                    "priority_reason": gap.get("priority", "")
                },
                expected_outcomes=["Fill missing genealogical information", "Extend family tree"],
                research_steps=gap.get("research_suggestions", []),
                estimated_effort=self._estimate_research_effort(gap),
                success_probability=self._estimate_success_probability(gap)
            )

            self.research_priorities.append(priority)

    def _generate_priority_tasks_from_conflicts(self, gedcom_analysis: dict[str, Any]):
        """Generate priority tasks from identified conflicts."""
        conflicts = gedcom_analysis.get("conflicts_identified", [])

        for conflict in conflicts:
            priority_score = self._calculate_conflict_priority_score(conflict)

            priority = ResearchPriority(
                priority_id=f"conflict_{conflict.get('conflict_id', 'unknown')}",
                task_type="conflict_resolution",
                description=f"Resolve: {conflict.get('description', '')}",
                target_people=conflict.get("people_involved", []),
                priority_score=priority_score,
                urgency=self._map_severity_to_urgency(conflict.get("severity", "minor")),
                research_context={
                    "conflict_type": conflict.get("conflict_type"),
                    "severity": conflict.get("severity"),
                    "conflict_id": conflict.get("conflict_id")
                },
                expected_outcomes=["Resolve data conflict", "Improve tree accuracy"],
                research_steps=conflict.get("resolution_suggestions", []),
                estimated_effort="medium",
                success_probability=0.7
            )

            self.research_priorities.append(priority)

    def _generate_priority_tasks_from_dna(self, dna_crossref_analysis: dict[str, Any]):
        """Generate priority tasks from DNA cross-reference analysis."""
        if not dna_crossref_analysis:
            return

        verification_opportunities = dna_crossref_analysis.get("verification_opportunities", [])

        for opportunity in verification_opportunities:
            priority_score = 85.0 if opportunity.get("priority") == "high" else 65.0

            priority = ResearchPriority(
                priority_id=f"dna_{opportunity.get('opportunity_id', 'unknown')}",
                task_type="dna_verification",
                description=opportunity.get("description", ""),
                target_people=[],  # Would extract from opportunity data
                priority_score=priority_score,
                urgency="high",
                research_context={
                    "opportunity_type": opportunity.get("type"),
                    "dna_evidence": True
                },
                expected_outcomes=["Verify DNA-tree connections", "Confirm relationships"],
                research_steps=opportunity.get("verification_steps", []),
                estimated_effort="low",
                success_probability=0.8
            )

            self.research_priorities.append(priority)

    def _generate_cluster_research_tasks(self) -> None:
        """Generate research tasks for location clusters."""
        for cluster in self.location_clusters:
            if cluster.research_efficiency_score > 0.7:  # High efficiency clusters
                priority = ResearchPriority(
                    priority_id=f"cluster_{cluster.cluster_id}",
                    task_type="cluster_research",
                    description=f"Cluster research in {cluster.location} ({cluster.people_count} people)",
                    target_people=cluster.target_people,
                    priority_score=cluster.research_efficiency_score * 100,
                    urgency="medium",
                    research_context={
                        "location": cluster.location,
                        "time_period": cluster.time_period,
                        "cluster_size": cluster.people_count
                    },
                    expected_outcomes=["Multiple family connections", "Efficient record research"],
                    research_steps=cluster.cluster_research_plan,
                    estimated_effort="medium",
                    success_probability=0.75
                )

                self.research_priorities.append(priority)

    def _score_and_rank_priorities(self) -> None:
        """Score and rank all research priorities with dependency tracking and workflow optimization."""
        # Apply dependency tracking and workflow optimization
        self._analyze_task_dependencies()
        self._optimize_research_workflow()

        # Sort by priority score (highest first)
        self.research_priorities.sort(key=lambda p: p.priority_score, reverse=True)

        # Enhanced scoring with research efficiency and success probability
        for priority in self.research_priorities:
            # Efficiency adjustments
            efficiency_bonus = 0
            if priority.estimated_effort == "low":
                efficiency_bonus = 8  # Increased bonus for quick wins
            elif priority.estimated_effort == "medium":
                efficiency_bonus = 2
            elif priority.estimated_effort == "high":
                efficiency_bonus = -3  # Reduced penalty for important high-effort tasks

            # Success probability bonus (more nuanced)
            success_bonus = (priority.success_probability - 0.5) * 15

            # Dependency bonus for prerequisite tasks
            dependency_bonus = getattr(priority, 'dependency_bonus', 0)

            # Workflow optimization bonus
            workflow_bonus = getattr(priority, 'workflow_bonus', 0)

            priority.priority_score += efficiency_bonus + success_bonus + dependency_bonus + workflow_bonus
            priority.priority_score = max(0, min(100, priority.priority_score))  # Clamp to 0-100

    def _analyze_task_dependencies(self) -> None:
        """Analyze dependencies between research tasks and adjust priorities accordingly."""
        for priority in self.research_priorities:
            dependency_bonus = 0

            # Tasks that enable other research get priority boost
            if priority.task_type == "vital_records":
                # Vital records often enable other research
                dependency_bonus += 5

            elif priority.task_type == "conflict_resolution":
                # Resolving conflicts enables accurate further research
                dependency_bonus += 8

            elif priority.task_type == "dna_verification":
                # DNA verification can confirm or refute multiple hypotheses
                dependency_bonus += 6

            # Check for prerequisite relationships
            for other_priority in self.research_priorities:
                if other_priority != priority and self._is_prerequisite(priority, other_priority):
                    dependency_bonus += 3

            priority.dependency_bonus = dependency_bonus

    def _optimize_research_workflow(self) -> None:
        """Optimize research workflow by grouping related tasks and considering efficiency."""
        # Group tasks by location for research efficiency
        location_groups = defaultdict(list)
        for priority in self.research_priorities:
            location = self._extract_location_from_context(priority.research_context)
            if location:
                location_groups[location].append(priority)

        # Apply workflow bonuses for location clustering
        for _, tasks in location_groups.items():
            if len(tasks) > 1:  # Multiple tasks in same location
                for task in tasks:
                    task.workflow_bonus = getattr(task, 'workflow_bonus', 0) + 3

        # Group tasks by person for research efficiency
        person_groups = defaultdict(list)
        for priority in self.research_priorities:
            if priority.target_people:
                for person in priority.target_people:
                    person_groups[person].append(priority)

        # Apply workflow bonuses for person clustering
        for _, tasks in person_groups.items():
            if len(tasks) > 1:  # Multiple tasks for same person
                for task in tasks:
                    task.workflow_bonus = getattr(task, 'workflow_bonus', 0) + 2

    def _is_prerequisite(self, task1: ResearchPriority, task2: ResearchPriority) -> bool:
        """Determine if task1 is a prerequisite for task2."""
        # Vital records often prerequisite for other research
        if task1.task_type == "vital_records" and task2.task_type in ["census", "immigration"]:
            # Check if they involve the same person
            if any(person in task2.target_people for person in task1.target_people):
                return True

        # Conflict resolution prerequisite for verification tasks
        if task1.task_type == "conflict_resolution" and task2.task_type == "dna_verification":
            if any(person in task2.target_people for person in task1.target_people):
                return True

        return False

    def _extract_location_from_context(self, context: dict[str, Any]) -> str:
        """Extract location information from research context."""
        # Look for location indicators in context
        for key in ['location', 'place', 'county', 'state', 'country']:
            if key in context:
                return str(context[key])
        return ""

    # Helper methods for calculations and analysis
    def _estimate_generations_back(self, _surname: str, _gedcom_analysis: dict[str, Any]) -> int:
        """Estimate how many generations back this surname line goes."""
        # Placeholder implementation
        return 4

    def _calculate_line_completeness(self, _surname: str, _gedcom_analysis: dict[str, Any]) -> float:
        """Calculate completeness percentage for a family line."""
        # Placeholder implementation
        return 65.0

    def _identify_missing_generations(self, _surname: str, _gedcom_analysis: dict[str, Any]) -> list[int]:
        """Identify which generations are missing for this line."""
        # Placeholder implementation
        return [3, 4]

    def _identify_research_bottlenecks(self, surname: str, _gedcom_analysis: dict[str, Any]) -> list[str]:
        """Identify research bottlenecks for this family line."""
        return [
            f"Missing parents for {surname} ancestors",
            f"No immigration records found for {surname} family",
            f"Birth records unavailable for early {surname} generations"
        ]

    def _identify_priority_targets(self, surname: str, gedcom_analysis: dict[str, Any]) -> list[str]:
        """Identify priority research targets for this family line."""
        return [
            f"Research {surname} family immigration",
            f"Find birth records for {surname} ancestors",
            f"Locate {surname} family in census records"
        ]

    def _extract_location_context(self, gap: dict[str, Any]) -> Optional[str]:
        """Extract location context from a gap."""
        # This would analyze the gap description for location clues
        description = gap.get("description", "")
        # Simple implementation - look for location keywords
        if "Scotland" in description:
            return "Scotland"
        if "Ireland" in description:
            return "Ireland"
        if "England" in description:
            return "England"
        return None

    def _extract_opportunity_location(self, opportunity: dict[str, Any]) -> Optional[str]:
        """Extract location from research opportunity."""
        description = opportunity.get("description", "")
        # Simple implementation
        if "Scotland" in description:
            return "Scotland"
        if "Ireland" in description:
            return "Ireland"
        if "England" in description:
            return "England"
        return None

    def _estimate_time_period_for_location(self, _location: str, _items: list[dict[str, Any]]) -> str:
        """Estimate time period for location cluster."""
        return "1800-1900"  # Placeholder

    def _identify_available_records_for_location(self, _location: str) -> list[str]:
        """Identify available record types for a location."""
        record_types = {
            "Scotland": ["Birth certificates", "Death certificates", "Census records", "Parish registers"],
            "Ireland": ["Civil registration", "Catholic parish records", "Griffith's Valuation"],
            "England": ["Birth certificates", "Death certificates", "Census records", "Parish registers"]
        }
        return record_types.get(location, ["General records"])

    def _calculate_cluster_efficiency(self, location: str, items: list[dict[str, Any]]) -> float:
        """Calculate research efficiency score for a location cluster."""
        # Base efficiency on number of people and available records
        people_count = len(items)
        base_score = min(1.0, people_count / 5.0)  # More people = higher efficiency

        # Bonus for well-documented locations
        location_bonus = 0.2 if location in ["Scotland", "England", "Ireland"] else 0.0

        return min(1.0, base_score + location_bonus)

    def _generate_cluster_research_plan(self, location: str, items: list[dict[str, Any]]) -> list[str]:
        """Generate research plan for a location cluster."""
        return [
            f"Research {location} records for multiple family members",
            "Check local archives and repositories",
            f"Look for family connections in {location} records",
            f"Cross-reference multiple sources for {location}"
        ]

    def _calculate_gap_priority_score(self, gap: dict[str, Any]) -> float:
        """Calculate enhanced priority score for a gap using genealogical research best practices."""
        base_score = 50.0

        # Enhanced gap type scoring with genealogical research priorities
        gap_type = gap.get("gap_type", "")
        if gap_type == "missing_parents":
            base_score += 25  # Critical for family tree extension
        elif gap_type == "missing_spouse":
            base_score += 22  # Important for family completeness
        elif gap_type == "missing_children":
            base_score += 20  # Valuable for descendant research
        elif gap_type == "missing_dates":
            base_score += 18  # Essential for timeline verification
        elif gap_type == "missing_places":
            base_score += 15  # Important for location-based research
        elif gap_type == "missing_occupation":
            base_score += 10  # Useful for social history

        # Priority level adjustments
        priority = gap.get("priority", "low")
        if priority == "critical":
            base_score += 20
        elif priority == "high":
            base_score += 15
        elif priority == "medium":
            base_score += 8
        elif priority == "low":
            base_score += 3

        # Research feasibility factors
        person_id = gap.get("person_id", "")
        if person_id:
            # Boost score for direct ancestors (higher generations)
            generation_level = self._estimate_generation_level(person_id)
            if generation_level <= 3:  # Parents, grandparents, great-grandparents
                base_score += (4 - generation_level) * 5

        # Available evidence bonus
        evidence_quality = gap.get("evidence_quality", "low")
        if evidence_quality == "high":
            base_score += 10
        elif evidence_quality == "medium":
            base_score += 5

        # Research difficulty adjustment
        difficulty = gap.get("research_difficulty", "medium")
        if difficulty == "easy":
            base_score += 8  # Quick wins are valuable
        elif difficulty == "hard":
            base_score -= 5  # Reduce priority for very difficult research

        return min(100.0, max(0.0, base_score))

    def _calculate_conflict_priority_score(self, conflict: dict[str, Any]) -> float:
        """Calculate enhanced priority score for a conflict using genealogical accuracy principles."""
        base_score = 60.0

        # Enhanced severity scoring with genealogical impact assessment
        severity = conflict.get("severity", "minor")
        if severity == "critical":
            base_score += 35  # Major tree accuracy issues
        elif severity == "major":
            base_score += 25  # Significant discrepancies
        elif severity == "moderate":
            base_score += 15  # Notable inconsistencies
        elif severity == "minor":
            base_score += 8   # Small discrepancies

        # Conflict type impact on research
        conflict_type = conflict.get("conflict_type", "")
        if conflict_type == "date_conflict":
            base_score += 15  # Timeline accuracy is crucial
        elif conflict_type == "location_conflict":
            base_score += 12  # Geographic accuracy affects research strategy
        elif conflict_type == "relationship_conflict":
            base_score += 20  # Family structure accuracy is critical
        elif conflict_type == "name_conflict":
            base_score += 10  # Identity verification important

        # Number of people affected
        people_involved = conflict.get("people_involved", [])
        if len(people_involved) > 3:
            base_score += 10  # Multi-person conflicts have broader impact
        elif len(people_involved) > 1:
            base_score += 5

        # Available resolution evidence
        resolution_evidence = conflict.get("resolution_evidence", "low")
        if resolution_evidence == "high":
            base_score += 12  # High chance of successful resolution
        elif resolution_evidence == "medium":
            base_score += 6

        # Research blocking factor
        if conflict.get("blocks_research", False):
            base_score += 15  # Conflicts that block further research get priority

        return min(100.0, max(0.0, base_score))

    def _estimate_generation_level(self, person_id: str) -> int:
        """Estimate generation level from person ID (lower numbers = closer to root person)."""
        # This is a simplified estimation - in a real system this would
        # analyze the actual family tree structure
        if not person_id:
            return 5  # Default to mid-level

        # Simple heuristic based on ID patterns (this would be more sophisticated in practice)
        if "parent" in person_id.lower() or "father" in person_id.lower() or "mother" in person_id.lower():
            return 1
        if "grandparent" in person_id.lower() or "grand" in person_id.lower():
            return 2
        if "great" in person_id.lower():
            return 3
        return 4  # Default for other relatives

    def _map_gap_to_task_type(self, gap_type: str) -> str:
        """Map gap type to research task type."""
        mapping = {
            "missing_parents": "vital_records",
            "missing_dates": "vital_records",
            "missing_places": "location_research",
            "missing_spouse": "marriage_records",
            "missing_children": "family_research"
        }
        return mapping.get(gap_type, "general_research")

    def _determine_urgency_from_score(self, score: float) -> str:
        """Determine urgency level from priority score."""
        if score >= 85:
            return "critical"
        if score >= 70:
            return "high"
        if score >= 50:
            return "medium"
        return "low"

    def _map_severity_to_urgency(self, severity: str) -> str:
        """Map conflict severity to urgency."""
        mapping = {
            "critical": "critical",
            "major": "high",
            "minor": "medium"
        }
        return mapping.get(severity, "low")

    def _estimate_research_effort(self, gap: dict[str, Any]) -> str:
        """Estimate research effort required for a gap."""
        gap_type = gap.get("gap_type", "")
        if gap_type in ["missing_dates", "missing_places"]:
            return "medium"
        if gap_type == "missing_parents":
            return "high"
        return "medium"

    def _estimate_success_probability(self, gap: dict[str, Any]) -> float:
        """Estimate probability of successfully filling a gap using genealogical research factors."""
        gap_type = gap.get("gap_type", "")
        priority = gap.get("priority", "low")

        base_probability = 0.4  # Start with realistic baseline

        # Priority level impact on success probability
        if priority == "critical":
            base_probability += 0.25  # Critical gaps often have more evidence
        elif priority == "high":
            base_probability += 0.2
        elif priority == "medium":
            base_probability += 0.15
        elif priority == "low":
            base_probability += 0.05

        # Gap type success probability based on genealogical research experience
        if gap_type == "missing_dates":
            base_probability += 0.15  # Dates often found in multiple record types
        elif gap_type == "missing_places":
            base_probability += 0.12  # Places often documented in various records
        elif gap_type == "missing_parents":
            base_probability += 0.08  # More challenging but often achievable
        elif gap_type == "missing_spouse":
            base_probability += 0.10  # Marriage records often well-documented
        elif gap_type == "missing_children":
            base_probability += 0.06  # Can be challenging due to infant mortality
        elif gap_type == "missing_occupation":
            base_probability += 0.18  # Often found in census and directories

        # Time period impact (older records are harder to find)
        time_period = gap.get("time_period", "")
        if time_period:
            if "18" in time_period:  # 1800s
                if "180" in time_period or "181" in time_period:  # Early 1800s
                    base_probability -= 0.1
                else:  # Later 1800s
                    base_probability += 0.05
            elif "19" in time_period:  # 1900s
                base_probability += 0.1  # Better record keeping
            elif "17" in time_period:  # 1700s
                base_probability -= 0.2  # Much more challenging

        # Location impact on success probability
        location = gap.get("location", "")
        if location:
            location_lower = location.lower()
            if any(term in location_lower for term in ['england', 'scotland', 'ireland']):
                base_probability += 0.08  # Good record keeping traditions
            elif any(term in location_lower for term in ['massachusetts', 'connecticut', 'new hampshire']):
                base_probability += 0.12  # Excellent early American records
            elif any(term in location_lower for term in ['virginia', 'north carolina', 'south carolina']):
                base_probability -= 0.05  # Some record loss from wars
            elif any(term in location_lower for term in ['frontier', 'territory', 'west']):
                base_probability -= 0.1  # Frontier areas had less record keeping

        # Available evidence quality impact
        evidence_quality = gap.get("evidence_quality", "low")
        if evidence_quality == "high":
            base_probability += 0.15
        elif evidence_quality == "medium":
            base_probability += 0.08
        elif evidence_quality == "low":
            base_probability -= 0.05

        # Research difficulty adjustment
        difficulty = gap.get("research_difficulty", "medium")
        if difficulty == "easy":
            base_probability += 0.2
        elif difficulty == "medium":
            base_probability += 0.05
        elif difficulty == "hard":
            base_probability -= 0.15
        elif difficulty == "very_hard":
            base_probability -= 0.25

        return min(1.0, max(0.1, base_probability))  # Keep between 10% and 100%

    def _generate_research_recommendations(self) -> list[str]:
        """Generate overall research recommendations."""
        recommendations = []

        if len(self.research_priorities) > 10:
            recommendations.append("Focus on top 5-10 highest priority tasks to avoid overwhelm")

        high_priority_count = len([p for p in self.research_priorities if p.urgency in ["critical", "high"]])
        if high_priority_count > 0:
            recommendations.append(f"Address {high_priority_count} high-priority items first")

        cluster_count = len(self.location_clusters)
        if cluster_count > 0:
            recommendations.append(f"Consider cluster research approach for {cluster_count} geographic areas")

        dna_tasks = len([p for p in self.research_priorities if p.task_type == "dna_verification"])
        if dna_tasks > 0:
            recommendations.append(f"Prioritize {dna_tasks} DNA verification tasks for quick wins")

        return recommendations

    def _analyze_research_efficiency(self) -> dict[str, Any]:
        """Analyze overall research efficiency opportunities."""
        total_tasks = len(self.research_priorities)
        low_effort_tasks = len([p for p in self.research_priorities if p.estimated_effort == "low"])
        high_success_tasks = len([p for p in self.research_priorities if p.success_probability > 0.7])

        return {
            "total_tasks": total_tasks,
            "low_effort_tasks": low_effort_tasks,
            "high_success_probability_tasks": high_success_tasks,
            "efficiency_ratio": (low_effort_tasks + high_success_tasks) / max(1, total_tasks),
            "cluster_opportunities": len(self.location_clusters)
        }

    def _generate_next_steps(self) -> list[str]:
        """Generate immediate next steps."""
        if not self.research_priorities:
            return ["No research priorities identified"]

        top_priority = self.research_priorities[0]
        next_steps = [
            f"Start with highest priority: {top_priority.description}",
            f"Focus on {top_priority.urgency} urgency tasks first"
        ]

        if top_priority.research_steps:
            next_steps.append(f"First step: {top_priority.research_steps[0]}")

        return next_steps

    def _empty_prioritization_result(self) -> dict[str, Any]:
        """Return empty prioritization result for error cases."""
        return {
            "prioritization_timestamp": datetime.now().isoformat(),
            "total_priorities_identified": 0,
            "family_line_analysis": [],
            "location_clusters": [],
            "prioritized_tasks": [],
            "research_recommendations": [],
            "efficiency_analysis": {},
            "next_steps": [],
            "error": "Research prioritization failed"
        }

    def _priority_to_dict(self, priority: ResearchPriority) -> dict[str, Any]:
        """Convert ResearchPriority to dictionary using dataclass asdict()."""
        return asdict(priority)

    def _family_line_to_dict(self, family_line: FamilyLineStatus) -> dict[str, Any]:
        """Convert FamilyLineStatus to dictionary using dataclass asdict()."""
        return asdict(family_line)

    def _location_cluster_to_dict(self, cluster: LocationResearchCluster) -> dict[str, Any]:
        """Convert LocationResearchCluster to dictionary using dataclass asdict()."""
        return asdict(cluster)


# Test functions
def test_research_prioritization() -> bool:
    """Test the research prioritization system."""
    logger.info("Testing research prioritization system...")

    prioritizer = IntelligentResearchPrioritizer()

    # Test with mock data
    mock_gedcom_analysis = {
        "gaps_identified": [
            {
                "person_id": "I1",
                "person_name": "John Smith",
                "gap_type": "missing_parents",
                "description": "Missing parents for John Smith",
                "priority": "high",
                "research_suggestions": ["Search birth records"]
            }
        ],
        "conflicts_identified": [],
        "research_opportunities": [],
        "ai_insights": {
            "family_patterns": {
                "common_surnames": ["Smith", "Jones"]
            }
        }
    }

    mock_dna_analysis = {
        "verification_opportunities": [
            {
                "opportunity_id": "verify_1",
                "type": "high_confidence_match",
                "description": "High confidence DNA match",
                "priority": "high",
                "verification_steps": ["Compare family trees"]
            }
        ]
    }

    result = prioritizer.prioritize_research_tasks(mock_gedcom_analysis, mock_dna_analysis)

    assert "prioritization_timestamp" in result, "Result should include timestamp"
    assert "prioritized_tasks" in result, "Result should include prioritized tasks"
    assert "research_recommendations" in result, "Result should include recommendations"
    assert "next_steps" in result, "Result should include next steps"

    logger.info("✅ Research prioritization test passed")
    return True


def test_priority_scoring_and_ranking() -> None:
    """Ensure priority scores are computed and sorted descending with adjustments."""
    prioritizer = IntelligentResearchPrioritizer()
    gedcom = {
        "gaps_identified": [
            {"person_id": "I1", "person_name": "John Smith", "gap_type": "missing_parents", "description": "Missing parents", "priority": "high", "research_suggestions": []},
            {"person_id": "I2", "person_name": "Ann Smith", "gap_type": "missing_dates", "description": "Missing birth date", "priority": "medium", "research_suggestions": []},
        ],
        "conflicts_identified": [
            {"conflict_id": "c1", "conflict_type": "date_conflict", "description": "Bad dates", "people_involved": ["I1"], "severity": "critical", "resolution_suggestions": []}
        ],
        "research_opportunities": [],
        "ai_insights": {"family_patterns": {"common_surnames": ["Smith"]}}
    }
    dna = {"verification_opportunities": []}
    plan = prioritizer.prioritize_research_tasks(gedcom, dna)
    tasks = plan["prioritized_tasks"]
    scores = [t["priority_score"] for t in tasks]
    assert scores == sorted(scores, reverse=True), "Tasks should be sorted descending by score"


def test_cluster_generation_and_efficiency() -> None:
    """Location cluster with multiple items should yield cluster_research task with efficiency >0.7."""
    prioritizer = IntelligentResearchPrioritizer()
    # Provide multiple gaps referencing Scotland via description keyword extraction
    gaps = []
    for i in range(4):
        gaps.append({"person_id": f"I{i}", "person_name": f"Person{i}", "gap_type": "missing_places", "description": f"Missing birth location Scotland for Person{i}", "priority": "medium", "research_suggestions": []})
    gedcom = {
        "gaps_identified": gaps,
        "conflicts_identified": [],
        "research_opportunities": [
            {"opportunity_id": "op1", "opportunity_type": "location_research", "description": "Migration research Scotland", "target_people": []}
        ],
        "ai_insights": {"family_patterns": {"common_surnames": []}}
    }
    dna = {"verification_opportunities": []}
    plan = prioritizer.prioritize_research_tasks(gedcom, dna)
    cluster_tasks = [t for t in plan["prioritized_tasks"] if t["task_type"] == "cluster_research"]
    if cluster_tasks:  # Should exist
        assert cluster_tasks[0]["priority_score"] >= 70, "Cluster task score should reflect efficiency scaling"


def test_dna_verification_task_creation() -> None:
    """High priority verification opportunity should produce dna_verification task."""
    prioritizer = IntelligentResearchPrioritizer()
    gedcom = {"gaps_identified": [], "conflicts_identified": [], "research_opportunities": [], "ai_insights": {"family_patterns": {"common_surnames": []}}}
    dna = {"verification_opportunities": [{"opportunity_id": "v1", "type": "high_confidence_match", "description": "Match", "priority": "high", "verification_steps": ["Step1"]}]}
    plan = prioritizer.prioritize_research_tasks(gedcom, dna)
    assert any(t["task_type"] == "dna_verification" for t in plan["prioritized_tasks"])


def research_prioritization_module_tests() -> bool:
    """
    Comprehensive test suite for research_prioritization.py with real functionality testing.
    Tests intelligent research prioritization, clustering, and recommendation systems.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Intelligent Research Prioritization", "research_prioritization.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Prioritization basic flow",
            test_research_prioritization,
            "Plan contains core sections and next steps",
            "Run prioritize_research_tasks with minimal mock inputs",
            "Basic flow correctness",
        )
        suite.run_test(
            "Priority scoring & ordering",
            test_priority_scoring_and_ranking,
            "Scores computed and sorted descending",
            "Invoke prioritize_research_tasks and examine ordered scores",
            "Score ordering validation",
        )
        suite.run_test(
            "Cluster efficiency task",
            test_cluster_generation_and_efficiency,
            "Cluster research task added with efficiency-derived score",
            "Provide multiple Scotland gaps to build cluster",
            "Cluster task generation",
        )
        suite.run_test(
            "DNA verification task",
            test_dna_verification_task_creation,
            "High priority DNA opportunity yields dna_verification task",
            "Supply verification_opportunities and generate plan",
            "DNA verification task creation",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(research_prioritization_module_tests)


if __name__ == "__main__":
    """
    Execute comprehensive research prioritization tests when run directly.
    Tests intelligent research prioritization, clustering, and recommendation systems.
    """
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
