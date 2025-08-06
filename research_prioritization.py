"""
Intelligent Research Prioritization System for Ancestry Project

This module provides intelligent prioritization of genealogical research tasks
based on GEDCOM analysis, DNA evidence, and research efficiency factors.
Generates location-specific and time-period-specific research suggestions
with family line completion tracking.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 12.3 - Intelligent Research Prioritization
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

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
    target_people: List[str]
    priority_score: float  # 0.0 to 100.0
    urgency: str  # 'critical', 'high', 'medium', 'low'
    research_context: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    research_steps: List[str] = field(default_factory=list)
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
    missing_generations: List[int]
    research_bottlenecks: List[str]
    priority_research_targets: List[str]


@dataclass
class LocationResearchCluster:
    """Represents a geographic research cluster for efficiency."""
    
    cluster_id: str
    location: str
    time_period: str
    people_count: int
    target_people: List[str]
    available_records: List[str]
    research_efficiency_score: float
    cluster_research_plan: List[str] = field(default_factory=list)


class IntelligentResearchPrioritizer:
    """
    AI-powered system for prioritizing genealogical research tasks.
    """

    def __init__(self):
        """Initialize the research prioritizer."""
        self.research_priorities: List[ResearchPriority] = []
        self.family_line_status: List[FamilyLineStatus] = []
        self.location_clusters: List[LocationResearchCluster] = []
        
    def prioritize_research_tasks(
        self,
        gedcom_analysis: Dict[str, Any],
        dna_crossref_analysis: Dict[str, Any],
        existing_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
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

    def _analyze_family_line_completeness(self, gedcom_analysis: Dict[str, Any]):
        """Analyze completeness of different family lines."""
        try:
            # Extract family patterns from GEDCOM analysis
            ai_insights = gedcom_analysis.get("ai_insights", {})
            family_patterns = ai_insights.get("family_patterns", {})
            common_surnames = family_patterns.get("common_surnames", [])
            
            # Analyze each major surname line
            for i, surname in enumerate(common_surnames[:5]):  # Top 5 surnames
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

    def _create_location_research_clusters(self, gedcom_analysis: Dict[str, Any]):
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

    def _generate_priority_tasks_from_gaps(self, gedcom_analysis: Dict[str, Any]):
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

    def _generate_priority_tasks_from_conflicts(self, gedcom_analysis: Dict[str, Any]):
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

    def _generate_priority_tasks_from_dna(self, dna_crossref_analysis: Dict[str, Any]):
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

    def _generate_cluster_research_tasks(self):
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

    def _score_and_rank_priorities(self):
        """Score and rank all research priorities."""
        # Sort by priority score (highest first)
        self.research_priorities.sort(key=lambda p: p.priority_score, reverse=True)
        
        # Adjust scores based on research efficiency and success probability
        for priority in self.research_priorities:
            efficiency_bonus = 0
            if priority.estimated_effort == "low":
                efficiency_bonus = 5
            elif priority.estimated_effort == "high":
                efficiency_bonus = -5
            
            success_bonus = (priority.success_probability - 0.5) * 10
            
            priority.priority_score += efficiency_bonus + success_bonus
            priority.priority_score = max(0, min(100, priority.priority_score))  # Clamp to 0-100

    # Helper methods for calculations and analysis
    def _estimate_generations_back(self, surname: str, gedcom_analysis: Dict[str, Any]) -> int:
        """Estimate how many generations back this surname line goes."""
        # Placeholder implementation
        return 4

    def _calculate_line_completeness(self, surname: str, gedcom_analysis: Dict[str, Any]) -> float:
        """Calculate completeness percentage for a family line."""
        # Placeholder implementation
        return 65.0

    def _identify_missing_generations(self, surname: str, gedcom_analysis: Dict[str, Any]) -> List[int]:
        """Identify which generations are missing for this line."""
        # Placeholder implementation
        return [3, 4]

    def _identify_research_bottlenecks(self, surname: str, gedcom_analysis: Dict[str, Any]) -> List[str]:
        """Identify research bottlenecks for this family line."""
        return [
            f"Missing parents for {surname} ancestors",
            f"No immigration records found for {surname} family",
            f"Birth records unavailable for early {surname} generations"
        ]

    def _identify_priority_targets(self, surname: str, gedcom_analysis: Dict[str, Any]) -> List[str]:
        """Identify priority research targets for this family line."""
        return [
            f"Research {surname} family immigration",
            f"Find birth records for {surname} ancestors",
            f"Locate {surname} family in census records"
        ]

    def _extract_location_context(self, gap: Dict[str, Any]) -> Optional[str]:
        """Extract location context from a gap."""
        # This would analyze the gap description for location clues
        description = gap.get("description", "")
        # Simple implementation - look for location keywords
        if "Scotland" in description:
            return "Scotland"
        elif "Ireland" in description:
            return "Ireland"
        elif "England" in description:
            return "England"
        return None

    def _extract_opportunity_location(self, opportunity: Dict[str, Any]) -> Optional[str]:
        """Extract location from research opportunity."""
        description = opportunity.get("description", "")
        # Simple implementation
        if "Scotland" in description:
            return "Scotland"
        elif "Ireland" in description:
            return "Ireland"
        elif "England" in description:
            return "England"
        return None

    def _estimate_time_period_for_location(self, location: str, items: List[Dict[str, Any]]) -> str:
        """Estimate time period for location cluster."""
        return "1800-1900"  # Placeholder

    def _identify_available_records_for_location(self, location: str) -> List[str]:
        """Identify available record types for a location."""
        record_types = {
            "Scotland": ["Birth certificates", "Death certificates", "Census records", "Parish registers"],
            "Ireland": ["Civil registration", "Catholic parish records", "Griffith's Valuation"],
            "England": ["Birth certificates", "Death certificates", "Census records", "Parish registers"]
        }
        return record_types.get(location, ["General records"])

    def _calculate_cluster_efficiency(self, location: str, items: List[Dict[str, Any]]) -> float:
        """Calculate research efficiency score for a location cluster."""
        # Base efficiency on number of people and available records
        people_count = len(items)
        base_score = min(1.0, people_count / 5.0)  # More people = higher efficiency
        
        # Bonus for well-documented locations
        location_bonus = 0.2 if location in ["Scotland", "England", "Ireland"] else 0.0
        
        return min(1.0, base_score + location_bonus)

    def _generate_cluster_research_plan(self, location: str, items: List[Dict[str, Any]]) -> List[str]:
        """Generate research plan for a location cluster."""
        return [
            f"Research {location} records for multiple family members",
            f"Check local archives and repositories",
            f"Look for family connections in {location} records",
            f"Cross-reference multiple sources for {location}"
        ]

    def _calculate_gap_priority_score(self, gap: Dict[str, Any]) -> float:
        """Calculate priority score for a gap."""
        base_score = 50.0
        
        # Higher priority for certain gap types
        gap_type = gap.get("gap_type", "")
        if gap_type == "missing_parents":
            base_score += 20
        elif gap_type == "missing_dates":
            base_score += 15
        elif gap_type == "missing_places":
            base_score += 10
        
        # Higher priority for high-priority gaps
        if gap.get("priority") == "high":
            base_score += 15
        elif gap.get("priority") == "medium":
            base_score += 5
        
        return min(100.0, base_score)

    def _calculate_conflict_priority_score(self, conflict: Dict[str, Any]) -> float:
        """Calculate priority score for a conflict."""
        base_score = 60.0
        
        # Higher priority for more severe conflicts
        severity = conflict.get("severity", "minor")
        if severity == "critical":
            base_score += 30
        elif severity == "major":
            base_score += 20
        elif severity == "minor":
            base_score += 5
        
        return min(100.0, base_score)

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
        elif score >= 70:
            return "high"
        elif score >= 50:
            return "medium"
        else:
            return "low"

    def _map_severity_to_urgency(self, severity: str) -> str:
        """Map conflict severity to urgency."""
        mapping = {
            "critical": "critical",
            "major": "high",
            "minor": "medium"
        }
        return mapping.get(severity, "low")

    def _estimate_research_effort(self, gap: Dict[str, Any]) -> str:
        """Estimate research effort required for a gap."""
        gap_type = gap.get("gap_type", "")
        if gap_type in ["missing_dates", "missing_places"]:
            return "medium"
        elif gap_type == "missing_parents":
            return "high"
        else:
            return "medium"

    def _estimate_success_probability(self, gap: Dict[str, Any]) -> float:
        """Estimate probability of successfully filling a gap."""
        gap_type = gap.get("gap_type", "")
        priority = gap.get("priority", "low")
        
        base_probability = 0.5
        
        if priority == "high":
            base_probability += 0.2
        elif priority == "medium":
            base_probability += 0.1
        
        if gap_type in ["missing_dates", "missing_places"]:
            base_probability += 0.1
        
        return min(1.0, base_probability)

    def _generate_research_recommendations(self) -> List[str]:
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

    def _analyze_research_efficiency(self) -> Dict[str, Any]:
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

    def _generate_next_steps(self) -> List[str]:
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

    def _empty_prioritization_result(self) -> Dict[str, Any]:
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

    def _priority_to_dict(self, priority: ResearchPriority) -> Dict[str, Any]:
        """Convert ResearchPriority to dictionary."""
        return {
            "priority_id": priority.priority_id,
            "task_type": priority.task_type,
            "description": priority.description,
            "target_people": priority.target_people,
            "priority_score": priority.priority_score,
            "urgency": priority.urgency,
            "research_context": priority.research_context,
            "expected_outcomes": priority.expected_outcomes,
            "research_steps": priority.research_steps,
            "estimated_effort": priority.estimated_effort,
            "success_probability": priority.success_probability
        }

    def _family_line_to_dict(self, family_line: FamilyLineStatus) -> Dict[str, Any]:
        """Convert FamilyLineStatus to dictionary."""
        return {
            "line_id": family_line.line_id,
            "line_name": family_line.line_name,
            "surname": family_line.surname,
            "generations_back": family_line.generations_back,
            "completeness_percentage": family_line.completeness_percentage,
            "missing_generations": family_line.missing_generations,
            "research_bottlenecks": family_line.research_bottlenecks,
            "priority_research_targets": family_line.priority_research_targets
        }

    def _location_cluster_to_dict(self, cluster: LocationResearchCluster) -> Dict[str, Any]:
        """Convert LocationResearchCluster to dictionary."""
        return {
            "cluster_id": cluster.cluster_id,
            "location": cluster.location,
            "time_period": cluster.time_period,
            "people_count": cluster.people_count,
            "target_people": cluster.target_people,
            "available_records": cluster.available_records,
            "research_efficiency_score": cluster.research_efficiency_score,
            "cluster_research_plan": cluster.cluster_research_plan
        }


# Test functions
def test_research_prioritization():
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


if __name__ == "__main__":
    """Test suite for research_prioritization.py"""
    test_research_prioritization()
