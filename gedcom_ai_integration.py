"""
GEDCOM AI Integration Module for Ancestry Project

This module integrates all Phase 12 components (GEDCOM Intelligence, DNA-GEDCOM
Cross-Reference, and Research Prioritization) into a unified system that can be
used by the existing action modules to enhance genealogical research.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 12 - Advanced GEDCOM Integration & Family Tree Intelligence
"""

from datetime import datetime
from typing import Any, Optional

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)

# === PHASE 12 IMPORTS ===
try:
    from dna_gedcom_crossref import DNAGedcomCrossReferencer, DNAMatch
    from gedcom_intelligence import GedcomIntelligenceAnalyzer
    from research_prioritization import IntelligentResearchPrioritizer
    GEDCOM_AI_AVAILABLE = True
    logger.info("GEDCOM AI integration components loaded successfully")
except ImportError as e:
    logger.warning(f"GEDCOM AI integration not available: {e}")
    GEDCOM_AI_AVAILABLE = False
    GedcomIntelligenceAnalyzer = None
    DNAGedcomCrossReferencer = None
    IntelligentResearchPrioritizer = None
    DNAMatch = None


class GedcomAIIntegrator:
    """
    Unified interface for all GEDCOM AI analysis capabilities.
    Integrates intelligence analysis, DNA cross-referencing, and research prioritization.
    """

    def __init__(self):
        """Initialize the GEDCOM AI integrator."""
        self.intelligence_analyzer = None
        self.dna_crossref = None
        self.research_prioritizer = None

        if GEDCOM_AI_AVAILABLE:
            try:
                if all([
                    GedcomIntelligenceAnalyzer is not None,
                    DNAGedcomCrossReferencer is not None,
                    IntelligentResearchPrioritizer is not None,
                ]):
                    self.intelligence_analyzer = GedcomIntelligenceAnalyzer()  # type: ignore[operator]
                    self.dna_crossref = DNAGedcomCrossReferencer()  # type: ignore[operator]
                    self.research_prioritizer = IntelligentResearchPrioritizer()  # type: ignore[operator]
                else:
                    raise RuntimeError("AI component classes not loaded (None)")
            except Exception as init_err:
                logger.warning(f"Failed initializing GEDCOM AI components, falling back: {init_err}")
                self.intelligence_analyzer = None
                self.dna_crossref = None
                self.research_prioritizer = None
                globals()["GEDCOM_AI_AVAILABLE"] = False
            logger.info("GEDCOM AI integrator initialized successfully")
        else:
            logger.warning("GEDCOM AI integrator initialized without AI components")

    def perform_comprehensive_analysis(
        self,
        gedcom_data: Any,
        dna_matches_data: Optional[list[dict[str, Any]]] = None,
        tree_owner_info: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Perform comprehensive GEDCOM AI analysis including intelligence, cross-referencing, and prioritization.

        Args:
            gedcom_data: GEDCOM data instance
            dna_matches_data: Optional DNA match data
            tree_owner_info: Optional tree owner information

        Returns:
            Comprehensive analysis results
        """
        try:
            if not GEDCOM_AI_AVAILABLE:
                return self._unavailable_analysis_result()

            logger.info("Starting comprehensive GEDCOM AI analysis")

            # Step 1: GEDCOM Intelligence Analysis
            logger.debug("Performing GEDCOM intelligence analysis...")
            if self.intelligence_analyzer is None:
                return self._unavailable_analysis_result()
            gedcom_analysis = self.intelligence_analyzer.analyze_gedcom_data(gedcom_data)

            # Step 2: DNA-GEDCOM Cross-Reference (if DNA data available)
            dna_analysis = {}
            if dna_matches_data:
                logger.debug("Performing DNA-GEDCOM cross-reference analysis...")
                if self.dna_crossref is not None:
                    dna_matches = self._convert_to_dna_matches(dna_matches_data)
                    dna_analysis = self.dna_crossref.analyze_dna_gedcom_connections(
                        dna_matches, gedcom_data, tree_owner_info
                    )

            # Step 3: Research Prioritization
            logger.debug("Performing research prioritization...")
            if self.research_prioritizer is None:
                return self._unavailable_analysis_result()
            prioritization_analysis = self.research_prioritizer.prioritize_research_tasks(
                gedcom_analysis, dna_analysis
            )

            # Combine all analyses
            comprehensive_result = {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": "comprehensive_gedcom_ai",
                "gedcom_intelligence": gedcom_analysis,
                "dna_crossref": dna_analysis,
                "research_prioritization": prioritization_analysis,
                "integrated_insights": self._generate_integrated_insights(
                    gedcom_analysis, dna_analysis, prioritization_analysis
                ),
                "actionable_recommendations": self._generate_actionable_recommendations(
                    gedcom_analysis, dna_analysis, prioritization_analysis
                ),
                "summary": self._generate_comprehensive_summary(
                    gedcom_analysis, dna_analysis, prioritization_analysis
                )
            }

            logger.info("Comprehensive GEDCOM AI analysis completed successfully")
            return comprehensive_result

        except Exception as e:
            logger.error(f"Error during comprehensive GEDCOM AI analysis: {e}")
            return self._error_analysis_result(str(e))

    def generate_enhanced_research_tasks(
        self,
        person_data: dict[str, Any],
        extracted_genealogical_data: dict[str, Any],
        gedcom_data: Any = None
    ) -> list[dict[str, Any]]:
        """
        Generate enhanced research tasks using GEDCOM AI analysis.
        This method can be called by existing action modules.

        Args:
            person_data: Information about the person being researched
            extracted_genealogical_data: Genealogical data extracted from conversations
            gedcom_data: Optional GEDCOM data for enhanced analysis

        Returns:
            List of enhanced research tasks
        """
        try:
            if not GEDCOM_AI_AVAILABLE:
                return self._fallback_research_tasks(person_data, extracted_genealogical_data)

            enhanced_tasks = []

            # If GEDCOM data is available, use AI analysis
            if gedcom_data:
                analysis = self.perform_comprehensive_analysis(gedcom_data)

                # Extract relevant tasks for this person
                prioritized_tasks = analysis.get("research_prioritization", {}).get("prioritized_tasks", [])

                # Filter tasks relevant to this person
                person_name = person_data.get("username", "")
                relevant_tasks = [
                    task for task in prioritized_tasks
                    if person_name in task.get("target_people", []) or
                       any(person_name.lower() in target.lower() for target in task.get("target_people", []))
                ]

                # Convert to enhanced task format
                for task in relevant_tasks[:3]:  # Top 3 relevant tasks
                    enhanced_task = {
                        "title": f"GEDCOM AI: {task.get('description', 'Research Task')}",
                        "description": self._format_enhanced_task_description(task),
                        "category": task.get("task_type", "general"),
                        "priority": task.get("urgency", "medium"),
                        "template_used": "gedcom_ai_enhanced"
                    }
                    enhanced_tasks.append(enhanced_task)

            # Add general AI-enhanced tasks based on extracted data
            general_tasks = self._generate_ai_enhanced_tasks_from_data(extracted_genealogical_data)
            enhanced_tasks.extend(general_tasks)

            return enhanced_tasks[:5]  # Limit to top 5 tasks

        except Exception as e:
            logger.error(f"Error generating enhanced research tasks: {e}")
            return self._fallback_research_tasks(person_data, extracted_genealogical_data)

    def get_gedcom_insights_for_person(
        self,
        person_identifier: str,
        gedcom_data: Any
    ) -> dict[str, Any]:
        """
        Get GEDCOM AI insights for a specific person.

        Args:
            person_identifier: Person ID or name to analyze
            gedcom_data: GEDCOM data instance

        Returns:
            Person-specific insights from GEDCOM AI analysis
        """
        try:
            if not GEDCOM_AI_AVAILABLE or not gedcom_data:
                return {"insights": "GEDCOM AI analysis not available"}

            # Perform analysis
            if self.intelligence_analyzer is None:
                return {"insights": "GEDCOM AI analysis not available"}
            analysis = self.intelligence_analyzer.analyze_gedcom_data(gedcom_data)

            # Extract person-specific insights
            return {
                "person_identifier": person_identifier,
                "relevant_gaps": self._find_person_relevant_gaps(person_identifier, analysis),
                "relevant_conflicts": self._find_person_relevant_conflicts(person_identifier, analysis),
                "research_opportunities": self._find_person_research_opportunities(person_identifier, analysis),
                "family_context": self._get_person_family_context(person_identifier, gedcom_data),
                "ai_recommendations": self._get_person_ai_recommendations(person_identifier, analysis)
            }


        except Exception as e:
            logger.error(f"Error getting GEDCOM insights for person {person_identifier}: {e}")
            return {"error": f"Failed to get insights: {e}"}

    def _convert_to_dna_matches(self, dna_matches_data: list[dict[str, Any]]) -> list[Any]:
        """Convert DNA match data to DNAMatch objects."""
        dna_matches = []

        for match_data in dna_matches_data:
            if DNAMatch is None:
                continue
            try:
                dna_match = DNAMatch(
                    match_id=match_data.get("match_id", "unknown"),
                    match_name=match_data.get("match_name", "Unknown"),
                    estimated_relationship=match_data.get("estimated_relationship", "unknown"),
                    shared_dna_cm=match_data.get("shared_dna_cm"),
                    testing_company=match_data.get("testing_company", "Ancestry"),
                    confidence_level=match_data.get("confidence_level", "medium"),
                    shared_ancestors=match_data.get("shared_ancestors", [])
                )
                dna_matches.append(dna_match)
            except Exception as e:
                logger.debug(f"Error converting DNA match data: {e}")
                continue

        return dna_matches

    def _generate_integrated_insights(
        self,
        gedcom_analysis: dict[str, Any],
        dna_analysis: dict[str, Any],
        prioritization_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate integrated insights from all analyses."""
        return {
            "tree_health_score": self._calculate_tree_health_score(gedcom_analysis),
            "research_efficiency_opportunities": self._identify_efficiency_opportunities(prioritization_analysis),
            "dna_verification_potential": self._assess_dna_verification_potential(dna_analysis),
            "priority_research_areas": self._identify_priority_research_areas(prioritization_analysis),
            "data_quality_assessment": self._assess_data_quality(gedcom_analysis)
        }


    def _generate_actionable_recommendations(
        self,
        gedcom_analysis: dict[str, Any],
        dna_analysis: dict[str, Any],
        prioritization_analysis: dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations from all analyses."""
        recommendations = []

        # From GEDCOM analysis
        gedcom_summary = gedcom_analysis.get("summary", {})
        high_priority_items = gedcom_summary.get("high_priority_items", 0)
        if high_priority_items > 0:
            recommendations.append(f"Address {high_priority_items} high-priority GEDCOM issues first")

        # From DNA analysis
        if dna_analysis:
            high_confidence_matches = dna_analysis.get("summary", {}).get("high_confidence_matches", 0)
            if high_confidence_matches > 0:
                recommendations.append(f"Verify {high_confidence_matches} high-confidence DNA-tree matches")

        # From prioritization analysis
        prioritization_recs = prioritization_analysis.get("research_recommendations", [])
        recommendations.extend(prioritization_recs[:3])  # Top 3 prioritization recommendations

        return recommendations[:5]  # Limit to top 5 recommendations

    def _generate_comprehensive_summary(
        self,
        gedcom_analysis: dict[str, Any],
        dna_analysis: dict[str, Any],
        prioritization_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive summary of all analyses."""
        summary = {
            "gedcom_individuals_analyzed": gedcom_analysis.get("individuals_analyzed", 0),
            "total_gaps_identified": gedcom_analysis.get("summary", {}).get("total_gaps", 0),
            "total_conflicts_identified": gedcom_analysis.get("summary", {}).get("total_conflicts", 0),
            "research_priorities_generated": prioritization_analysis.get("total_priorities_identified", 0),
            "analysis_completeness": "comprehensive" if dna_analysis else "gedcom_only"
        }

        if dna_analysis:
            summary.update({
                "dna_matches_analyzed": dna_analysis.get("dna_matches_analyzed", 0),
                "dna_crossref_matches": dna_analysis.get("summary", {}).get("total_cross_references", 0)
            })

        return summary

    def _format_enhanced_task_description(self, task: dict[str, Any]) -> str:
        """Format enhanced task description with AI insights."""
        description = task.get("description", "")
        priority_score = task.get("priority_score", 0)
        success_probability = task.get("success_probability", 0.5)

        enhanced_description = f"{description}\n\n"
        enhanced_description += f"Priority Score: {priority_score:.1f}/100\n"
        enhanced_description += f"Success Probability: {success_probability:.0%}\n"
        enhanced_description += f"Estimated Effort: {task.get('estimated_effort', 'medium').title()}\n\n"

        research_steps = task.get("research_steps", [])
        if research_steps:
            enhanced_description += "AI-Recommended Steps:\n"
            for i, step in enumerate(research_steps[:3], 1):
                enhanced_description += f"{i}. {step}\n"

        expected_outcomes = task.get("expected_outcomes", [])
        if expected_outcomes:
            enhanced_description += f"\nExpected Outcomes: {', '.join(expected_outcomes)}"

        return enhanced_description

    def _generate_ai_enhanced_tasks_from_data(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate AI-enhanced tasks from extracted genealogical data."""
        tasks = []

        # Enhanced tasks based on structured names
        structured_names = extracted_data.get("structured_names", [])
        if structured_names:
            task = {
                "title": "AI-Enhanced Name Research",
                "description": f"Research {len(structured_names)} individuals using AI-powered genealogical analysis",
                "category": "ai_enhanced",
                "priority": "medium",
                "template_used": "ai_enhanced_names"
            }
            tasks.append(task)

        # Enhanced tasks based on locations
        locations = extracted_data.get("locations", [])
        if locations:
            task = {
                "title": "AI-Enhanced Location Research",
                "description": f"Analyze {len(locations)} locations using AI geographic and temporal analysis",
                "category": "ai_enhanced",
                "priority": "medium",
                "template_used": "ai_enhanced_locations"
            }
            tasks.append(task)

        return tasks

    def _fallback_research_tasks(self, person_data: dict[str, Any], extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate fallback research tasks when GEDCOM AI is not available."""
        return [
            {
                "title": f"Research: {person_data.get('username', 'Unknown')}",
                "description": "Standard genealogical research task (GEDCOM AI not available)",
                "category": "general",
                "priority": "medium",
                "template_used": "fallback"
            }
        ]

    # Helper methods for person-specific insights
    def _find_person_relevant_gaps(self, person_identifier: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Find gaps relevant to a specific person."""
        gaps = analysis.get("gaps_identified", [])
        return [gap for gap in gaps if person_identifier.lower() in gap.get("person_name", "").lower()]

    def _find_person_relevant_conflicts(self, person_identifier: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Find conflicts relevant to a specific person."""
        conflicts = analysis.get("conflicts_identified", [])
        return [conflict for conflict in conflicts if person_identifier in conflict.get("people_involved", [])]

    def _find_person_research_opportunities(self, person_identifier: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Find research opportunities relevant to a specific person."""
        opportunities = analysis.get("research_opportunities", [])
        return [opp for opp in opportunities if person_identifier in opp.get("target_people", [])]

    def _get_person_family_context(self, person_identifier: str, gedcom_data: Any) -> dict[str, Any]:
        """Get family context for a specific person."""
        # This would extract family relationships and context
        return {"family_context": "Analysis not yet implemented"}

    def _get_person_ai_recommendations(self, person_identifier: str, analysis: dict[str, Any]) -> list[str]:
        """Get AI recommendations for a specific person."""
        return [
            f"Focus on high-priority research for {person_identifier}",
            "Use DNA evidence to verify uncertain relationships",
            "Consider cluster research approach for efficiency"
        ]

    # Helper methods for integrated insights
    def _calculate_tree_health_score(self, gedcom_analysis: dict[str, Any]) -> float:
        """Calculate overall tree health score."""
        summary = gedcom_analysis.get("summary", {})
        total_gaps = summary.get("total_gaps", 0)
        total_conflicts = summary.get("total_conflicts", 0)
        individuals = gedcom_analysis.get("individuals_analyzed", 1)

        # Simple health score calculation
        issue_ratio = (total_gaps + total_conflicts) / max(1, individuals)
        health_score = max(0, 100 - (issue_ratio * 20))

        return round(health_score, 1)

    def _identify_efficiency_opportunities(self, prioritization_analysis: dict[str, Any]) -> list[str]:
        """Identify research efficiency opportunities."""
        efficiency = prioritization_analysis.get("efficiency_analysis", {})
        opportunities = []

        cluster_count = efficiency.get("cluster_opportunities", 0)
        if cluster_count > 0:
            opportunities.append(f"Cluster research in {cluster_count} geographic areas")

        low_effort_tasks = efficiency.get("low_effort_tasks", 0)
        if low_effort_tasks > 0:
            opportunities.append(f"Focus on {low_effort_tasks} low-effort, high-impact tasks")

        return opportunities

    def _assess_dna_verification_potential(self, dna_analysis: dict[str, Any]) -> str:
        """Assess DNA verification potential."""
        if not dna_analysis:
            return "No DNA data available"

        verification_opportunities = len(dna_analysis.get("verification_opportunities", []))
        if verification_opportunities > 5:
            return "High DNA verification potential"
        if verification_opportunities > 2:
            return "Moderate DNA verification potential"
        return "Limited DNA verification potential"

    def _identify_priority_research_areas(self, prioritization_analysis: dict[str, Any]) -> list[str]:
        """Identify priority research areas."""
        prioritized_tasks = prioritization_analysis.get("prioritized_tasks", [])

        # Group by task type
        task_types = {}
        for task in prioritized_tasks[:10]:  # Top 10 tasks
            task_type = task.get("task_type", "general")
            task_types[task_type] = task_types.get(task_type, 0) + 1

        # Return top task types
        sorted_types = sorted(task_types.items(), key=lambda x: x[1], reverse=True)
        return [task_type for task_type, count in sorted_types[:3]]

    def _assess_data_quality(self, gedcom_analysis: dict[str, Any]) -> str:
        """Assess overall data quality."""
        summary = gedcom_analysis.get("summary", {})
        total_conflicts = summary.get("total_conflicts", 0)
        individuals = gedcom_analysis.get("individuals_analyzed", 1)

        conflict_ratio = total_conflicts / max(1, individuals)

        if conflict_ratio < 0.05:
            return "High data quality"
        if conflict_ratio < 0.15:
            return "Good data quality"
        if conflict_ratio < 0.30:
            return "Fair data quality"
        return "Poor data quality - needs attention"

    def _unavailable_analysis_result(self) -> dict[str, Any]:
        """Return result when GEDCOM AI is not available."""
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_type": "unavailable",
            "error": "GEDCOM AI components not available",
            "gedcom_intelligence": {},
            "dna_crossref": {},
            "research_prioritization": {},
            "integrated_insights": {},
            "actionable_recommendations": ["Install GEDCOM AI components for enhanced analysis"],
            "summary": {"status": "unavailable"}
        }

    def _error_analysis_result(self, error_message: str) -> dict[str, Any]:
        """Return error result."""
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_type": "error",
            "error": error_message,
            "gedcom_intelligence": {},
            "dna_crossref": {},
            "research_prioritization": {},
            "integrated_insights": {},
            "actionable_recommendations": [],
            "summary": {"status": "error"}
        }


# Test functions
def test_gedcom_ai_integration():
    """Test the GEDCOM AI integration system."""
    try:
        # Simple test that doesn't rely on complex imports
        integrator = GedcomAIIntegrator()

        # Test basic instantiation
        assert integrator is not None, "Integrator should instantiate"

        # Test availability flag
        assert isinstance(GEDCOM_AI_AVAILABLE, bool), "GEDCOM_AI_AVAILABLE should be boolean"

        # Test basic functionality without complex operations
        if GEDCOM_AI_AVAILABLE:
            # Test that components are available
            assert integrator.intelligence_analyzer is not None, "Intelligence analyzer should be available"
            assert integrator.dna_crossref is not None, "DNA crossref should be available"
            assert integrator.research_prioritizer is not None, "Research prioritizer should be available"
        else:
            # Test that components are None when not available
            assert integrator.intelligence_analyzer is None, "Intelligence analyzer should be None"
            assert integrator.dna_crossref is None, "DNA crossref should be None"
            assert integrator.research_prioritizer is None, "Research prioritizer should be None"

        return True
    except Exception as e:
        print(f"GEDCOM AI integration test failed: {e}")
        return False


def test_fallback_research_tasks():
    """When components unavailable, fallback tasks should be returned."""
    integrator = GedcomAIIntegrator()
    # Force fallback by simulating unavailable components
    if not GEDCOM_AI_AVAILABLE:
        tasks = integrator.generate_enhanced_research_tasks({"username": "Tester"}, {"structured_names": []}, gedcom_data=None)
        assert tasks and tasks[0]["template_used"] == "fallback"
    else:
        # Temporarily monkey patch availability to simulate fallback
        original_flag = globals().get("GEDCOM_AI_AVAILABLE")
        globals()["GEDCOM_AI_AVAILABLE"] = False
        try:
            tasks = integrator.generate_enhanced_research_tasks({"username": "Tester"}, {"structured_names": []}, gedcom_data=None)
            assert tasks and tasks[0]["template_used"] == "fallback"
        finally:
            globals()["GEDCOM_AI_AVAILABLE"] = original_flag


def test_enhanced_task_description_formatting():
    """Ensure enhanced task description includes key metrics and steps."""
    integrator = GedcomAIIntegrator()
    sample_task = {
        "description": "Resolve data conflict",
        "priority_score": 72.5,
        "success_probability": 0.65,
        "estimated_effort": "low",
        "research_steps": ["Check census", "Locate vital record", "Compare trees"],
        "expected_outcomes": ["Resolved conflict"]
    }
    desc = integrator._format_enhanced_task_description(sample_task)
    for token in ["Priority Score", "Success Probability", "AI-Recommended Steps", "Expected Outcomes"]:
        assert token in desc, f"Description missing {token}"


def test_comprehensive_analysis_structure_with_stubs():
    """Test perform_comprehensive_analysis returns combined structure using stub analyzers when available."""
    if not GEDCOM_AI_AVAILABLE:
        # Skip if components not available; availability is already tested elsewhere
        return True
    integrator = GedcomAIIntegrator()

    # Create minimal stub gedcom data object
    stub = type('StubGedcom', (), {
        'indi_index': {'I1': type('Person', (), {'name': ['Jane Doe']})()},
        'id_to_parents': {},
        'id_to_children': {}
    })()

    result = integrator.perform_comprehensive_analysis(stub, dna_matches_data=[{"match_id": "m1", "match_name": "Jane Doe", "estimated_relationship": "2nd cousin", "shared_dna_cm": 150}], tree_owner_info={})
    for key in ["gedcom_intelligence", "dna_crossref", "research_prioritization", "integrated_insights", "summary"]:
        assert key in result, f"Comprehensive analysis missing {key}"
    assert result["summary"].get("analysis_completeness") in ["comprehensive", "gedcom_only"]


def test_person_specific_insights_fallback():
    """Person insights should provide indicative structure even with minimal data."""
    integrator = GedcomAIIntegrator()
    if not GEDCOM_AI_AVAILABLE:
        insights = integrator.get_gedcom_insights_for_person("Jane", None)
        assert "insights" in insights or "error" in insights
        return True
    stub = type('StubGedcom', (), {
        'indi_index': {'I1': type('Person', (), {'name': ['Jane Example']})()},
        'id_to_parents': {},
        'id_to_children': {}
    })()
    insights = integrator.get_gedcom_insights_for_person("Jane", stub)
    for key in ["relevant_gaps", "relevant_conflicts", "research_opportunities", "family_context", "ai_recommendations"]:
        assert key in insights


def gedcom_ai_integration_module_tests() -> bool:
    """
    Comprehensive test suite for gedcom_ai_integration.py with real functionality testing.
    Tests GEDCOM AI integration, analysis, and intelligent processing systems.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("GEDCOM AI Integration & Analysis", "gedcom_ai_integration.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Integration availability",
            test_gedcom_ai_integration,
            "Integrator instantiates and components reflect availability flag",
            "Instantiate GedcomAIIntegrator and validate component presence",
            "Validate basic integrator availability state",
        )
        suite.run_test(
            "Fallback research tasks",
            test_fallback_research_tasks,
            "Fallback task returned when components unavailable",
            "Force unavailable state and generate tasks",
            "Validate _fallback_research_tasks path",
        )
        suite.run_test(
            "Enhanced task description formatting",
            test_enhanced_task_description_formatting,
            "Description includes metrics, steps, and outcomes",
            "Format a synthetic priority task",
            "Validate _format_enhanced_task_description output",
        )
        suite.run_test(
            "Comprehensive analysis structure",
            test_comprehensive_analysis_structure_with_stubs,
            "Combined analysis includes required sections",
            "Run perform_comprehensive_analysis with stub data",
            "Validate comprehensive analysis keys",
        )
        suite.run_test(
            "Person-specific insights",
            test_person_specific_insights_fallback,
            "Person insight call returns structured keys",
            "Invoke get_gedcom_insights_for_person with stub",
            "Validate person insights extraction",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive GEDCOM AI integration tests using standardized TestSuite format."""
    return gedcom_ai_integration_module_tests()


if __name__ == "__main__":
    """
    Execute comprehensive GEDCOM AI integration tests when run directly.
    Tests GEDCOM AI integration, analysis, and intelligent processing systems.
    """
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
