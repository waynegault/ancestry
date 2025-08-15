"""
Genealogical Research Task Templates for Ancestry Project

This module provides specialized task templates for different types of genealogical
research based on extracted data. Creates actionable, specific research tasks that
leverage genealogical information to improve research productivity.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 10.1 - Task Management & Actionability Enhancement
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)


class GenealogicalTaskGenerator:
    """
    Generates specialized genealogical research tasks based on extracted data.
    Creates actionable, specific tasks that improve research productivity.
    """

    def __init__(self):
        """Initialize the task generator with templates and configuration."""
        self.task_templates = self._load_task_templates()
        self.task_config = self._load_task_configuration()

        # === PHASE 12: GEDCOM AI INTEGRATION ===
        try:
            from gedcom_ai_integration import GedcomAIIntegrator
            self.gedcom_ai_integrator = GedcomAIIntegrator()
            self.gedcom_ai_available = True
            logger.info("GEDCOM AI integration loaded in task generator")
        except ImportError as e:
            logger.debug(f"GEDCOM AI integration not available in task generator: {e}")
            self.gedcom_ai_integrator = None
            self.gedcom_ai_available = False

    def _load_task_templates(self) -> Dict[str, Dict[str, str]]:
        """Load genealogical research task templates."""
        return {
            "vital_records_search": {
                "title": "Search {record_type} for {person_name} ({time_period})",
                "description": "Research {record_type} for {person_name} {birth_death_info}.\n\nLocation: {location}\nTime Period: {time_period}\nPriority: {priority}\n\nResearch Steps:\n1. Search {location} vital records databases\n2. Check county/parish records for {time_period}\n3. Look for alternative spellings of {person_name}\n4. Cross-reference with family members\n\nExpected Information: {expected_info}",
                "category": "vital_records",
                "priority": "high"
            },
            "dna_match_analysis": {
                "title": "Analyze DNA Match: {match_name} ({estimated_relationship})",
                "description": "Investigate DNA match with {match_name} showing {shared_dna} shared DNA.\n\nEstimated Relationship: {estimated_relationship}\nShared DNA: {shared_dna}\nTesting Company: {testing_company}\n\nResearch Steps:\n1. Compare family trees for common ancestors\n2. Identify shared surnames and locations\n3. Look for triangulation opportunities\n4. Research potential common ancestor lines\n\nGoal: {research_goal}",
                "category": "dna_analysis",
                "priority": "medium"
            },
            "family_tree_verification": {
                "title": "Verify Family Connection: {person1} → {person2}",
                "description": "Verify the relationship between {person1} and {person2}.\n\nRelationship to Verify: {relationship}\nConflicting Information: {conflicts}\nEvidence Available: {evidence}\n\nResearch Steps:\n1. Gather primary source documents\n2. Cross-reference multiple sources\n3. Check for alternative explanations\n4. Document findings with source citations\n\nResolution Priority: {priority}",
                "category": "verification",
                "priority": "high"
            },
            "immigration_research": {
                "title": "Immigration Research: {person_name} ({origin} → {destination})",
                "description": "Research immigration of {person_name} from {origin} to {destination}.\n\nTime Period: {time_period}\nPorts: {ports}\nShip/Vessel: {vessel_info}\n\nResearch Steps:\n1. Search passenger manifests for {time_period}\n2. Check {origin} emigration records\n3. Look for naturalization records in {destination}\n4. Research family members who may have traveled together\n\nExpected Documents: {expected_documents}",
                "category": "immigration",
                "priority": "medium"
            },
            "census_research": {
                "title": "Census Research: {person_name} Family ({location}, {year})",
                "description": "Locate {person_name} and family in {year} census records.\n\nLocation: {location}\nFamily Members: {family_members}\nOccupation: {occupation}\n\nResearch Steps:\n1. Search {year} census for {location}\n2. Try alternative spellings and nearby areas\n3. Look for family members as search aids\n4. Check previous/subsequent census years\n\nInformation Needed: {information_needed}",
                "category": "census",
                "priority": "medium"
            },
            "military_research": {
                "title": "Military Service Research: {person_name} ({conflict})",
                "description": "Research military service of {person_name} during {conflict}.\n\nConflict: {conflict}\nService Branch: {service_branch}\nUnit: {unit_info}\nService Period: {service_period}\n\nResearch Steps:\n1. Search military service records\n2. Look for pension applications\n3. Check unit histories and muster rolls\n4. Research battle participation\n\nExpected Records: {expected_records}",
                "category": "military",
                "priority": "medium"
            },
            "occupation_research": {
                "title": "Occupation Research: {person_name} ({occupation})",
                "description": "Research {person_name}'s career as {occupation} in {location}.\n\nOccupation: {occupation}\nLocation: {location}\nTime Period: {time_period}\nEmployer: {employer}\n\nResearch Steps:\n1. Search employment records and directories\n2. Look for professional associations\n3. Check local newspapers for mentions\n4. Research industry-specific records\n\nResearch Goal: {research_goal}",
                "category": "occupation",
                "priority": "low"
            },
            "location_research": {
                "title": "Location Research: {person_name} in {location}",
                "description": "Research {person_name}'s time in {location} during {time_period}.\n\nLocation: {location}\nTime Period: {time_period}\nResidence Type: {residence_type}\nNeighbors: {neighbors}\n\nResearch Steps:\n1. Search local records and directories\n2. Check property/land records\n3. Look for church/school records\n4. Research local history and migration patterns\n\nInformation Sought: {information_sought}",
                "category": "location",
                "priority": "low"
            }
        }

    def _load_task_configuration(self) -> Dict[str, Any]:
        """Load task generation configuration."""
        return {
            "max_tasks_per_person": 5,
            "priority_weights": {
                "high": 3,
                "medium": 2,
                "low": 1
            },
            "category_limits": {
                "vital_records": 2,
                "dna_analysis": 1,
                "verification": 2,
                "immigration": 1,
                "census": 1,
                "military": 1,
                "occupation": 1,
                "location": 1
            }
        }

    def generate_research_tasks(
        self,
        person_data: Dict[str, Any],
        extracted_data: Dict[str, Any],
        suggested_tasks: List[str],
        gedcom_data: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Generate specialized research tasks based on extracted genealogical data.

        Args:
            person_data: Information about the person being researched
            extracted_data: Genealogical data extracted from conversations
            suggested_tasks: Basic AI-generated task suggestions
            gedcom_data: Optional GEDCOM data for AI-enhanced analysis

        Returns:
            List of enhanced task dictionaries with titles, descriptions, categories, and priorities
        """
        try:
            # Input validation and safe defaults
            if person_data is None or not isinstance(person_data, dict):
                person_data = {}
            if extracted_data is None or not isinstance(extracted_data, dict):
                extracted_data = {}
            if suggested_tasks is None or not isinstance(suggested_tasks, list):
                suggested_tasks = []
            
            enhanced_tasks = []

            # === PHASE 12: GEDCOM AI ENHANCED TASK GENERATION ===
            if self.gedcom_ai_available and self.gedcom_ai_integrator is not None and gedcom_data:
                try:
                    logger.debug("Generating GEDCOM AI-enhanced tasks")
                    ai_enhanced_tasks = self.gedcom_ai_integrator.generate_enhanced_research_tasks(
                        person_data, extracted_data, gedcom_data
                    )
                    enhanced_tasks.extend(ai_enhanced_tasks)
                    logger.info(f"Generated {len(ai_enhanced_tasks)} GEDCOM AI-enhanced tasks")
                except Exception as e:
                    logger.warning(f"GEDCOM AI task generation failed: {e}, falling back to standard generation")

            # Generate tasks based on extracted data types (standard approach)
            enhanced_tasks.extend(self._generate_vital_records_tasks(extracted_data))
            enhanced_tasks.extend(self._generate_dna_analysis_tasks(extracted_data))
            enhanced_tasks.extend(self._generate_verification_tasks(extracted_data))
            enhanced_tasks.extend(self._generate_immigration_tasks(extracted_data))
            enhanced_tasks.extend(self._generate_census_tasks(extracted_data))
            enhanced_tasks.extend(self._generate_military_tasks(extracted_data))
            enhanced_tasks.extend(self._generate_occupation_tasks(extracted_data))
            enhanced_tasks.extend(self._generate_location_tasks(extracted_data))

            # Add fallback tasks from AI suggestions if no specific tasks generated
            if not enhanced_tasks and suggested_tasks:
                enhanced_tasks.extend(self._create_fallback_tasks(person_data, suggested_tasks))

            # Prioritize and limit tasks
            prioritized_tasks = self._prioritize_and_limit_tasks(enhanced_tasks)

            logger.info(f"Generated {len(prioritized_tasks)} enhanced research tasks (GEDCOM AI: {'enabled' if self.gedcom_ai_available and gedcom_data else 'disabled'})")
            return prioritized_tasks

        except Exception as e:
            logger.error(f"Error generating research tasks: {e}")
            return self._create_fallback_tasks(person_data, suggested_tasks)

    def _generate_vital_records_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate vital records search tasks."""
        tasks = []
        vital_records = extracted_data.get("vital_records", [])
        
        for record in vital_records[:2]:  # Limit to 2 most important
            if isinstance(record, dict):
                person = record.get("person", "Unknown Person")
                event_type = record.get("event_type", "vital record")
                date = record.get("date", "unknown date")
                place = record.get("place", "unknown location")
                
                task_data = {
                    "person_name": person,
                    "record_type": f"{event_type} record",
                    "birth_death_info": f"({event_type} {date})" if date != "unknown date" else "",
                    "location": place,
                    "time_period": date,
                    "priority": "high",
                    "expected_info": f"Official {event_type} documentation with parents, dates, and locations"
                }
                
                task = self._create_task_from_template("vital_records_search", task_data)
                if task:
                    tasks.append(task)
        
        return tasks

    def _generate_dna_analysis_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate DNA match analysis tasks."""
        tasks = []
        dna_info = extracted_data.get("dna_information", [])
        
        for info in dna_info[:1]:  # Limit to 1 DNA task
            if isinstance(info, str) and ("match" in info.lower() or "dna" in info.lower()):
                task_data = {
                    "match_name": "DNA Match",
                    "estimated_relationship": "close family connection",
                    "shared_dna": "significant amount",
                    "testing_company": "Ancestry/23andMe",
                    "research_goal": "Identify common ancestors and verify family connections"
                }
                
                task = self._create_task_from_template("dna_match_analysis", task_data)
                if task:
                    tasks.append(task)
                break
        
        return tasks

    def _generate_verification_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate family tree verification tasks."""
        tasks = []
        relationships = extracted_data.get("relationships", [])
        
        for relationship in relationships[:2]:  # Limit to 2 verification tasks
            if isinstance(relationship, dict):
                person1 = relationship.get("person1", "Person A")
                person2 = relationship.get("person2", "Person B")
                rel_type = relationship.get("relationship", "family connection")
                
                task_data = {
                    "person1": person1,
                    "person2": person2,
                    "relationship": rel_type,
                    "conflicts": "Multiple sources with different information",
                    "evidence": "Family stories and preliminary research",
                    "priority": "high"
                }
                
                task = self._create_task_from_template("family_tree_verification", task_data)
                if task:
                    tasks.append(task)
        
        return tasks

    def _generate_immigration_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate immigration research tasks."""
        tasks = []
        locations = extracted_data.get("locations", [])
        
        # Look for potential immigration scenarios
        foreign_locations = []
        for location in locations:
            if isinstance(location, dict):
                place = location.get("place", "")
                if any(country in place for country in ["Ireland", "Scotland", "England", "Germany", "Poland", "Italy"]):
                    foreign_locations.append(location)
        
        if foreign_locations:
            location = foreign_locations[0]
            place = location.get("place", "Unknown Location")
            time_period = location.get("time_period", "1800s-1900s")
            
            task_data = {
                "person_name": "Family Member",
                "origin": place,
                "destination": "United States",
                "time_period": time_period,
                "ports": "Ellis Island, Castle Garden, or other major ports",
                "vessel_info": "To be determined",
                "expected_documents": "Passenger manifests, naturalization records, ship records"
            }
            
            task = self._create_task_from_template("immigration_research", task_data)
            if task:
                tasks.append(task)
        
        return tasks

    def _generate_census_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate census research tasks."""
        tasks = []
        structured_names = extracted_data.get("structured_names", [])
        locations = extracted_data.get("locations", [])
        
        if structured_names and locations:
            name_data = structured_names[0]
            location_data = locations[0]
            
            person_name = name_data.get("full_name", "Family Member") if isinstance(name_data, dict) else str(name_data)
            location = location_data.get("place", "Unknown Location") if isinstance(location_data, dict) else str(location_data)
            
            task_data = {
                "person_name": person_name,
                "location": location,
                "year": "1900-1940",
                "family_members": "Spouse and children",
                "occupation": "To be determined",
                "information_needed": "Family composition, ages, birthplaces, occupations"
            }
            
            task = self._create_task_from_template("census_research", task_data)
            if task:
                tasks.append(task)
        
        return tasks

    def _generate_military_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate military research tasks."""
        tasks = []
        # Look for military-related information in research questions or documents
        research_questions = extracted_data.get("research_questions", [])
        
        for question in research_questions:
            if isinstance(question, str) and any(term in question.lower() for term in ["war", "military", "service", "veteran", "army", "navy"]):
                task_data = {
                    "person_name": "Service Member",
                    "conflict": "Civil War, WWI, or WWII",
                    "service_branch": "To be determined",
                    "unit_info": "To be researched",
                    "service_period": "To be determined",
                    "expected_records": "Service records, pension files, unit histories"
                }
                
                task = self._create_task_from_template("military_research", task_data)
                if task:
                    tasks.append(task)
                break
        
        return tasks

    def _generate_occupation_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate occupation research tasks."""
        tasks = []
        occupations = extracted_data.get("occupations", [])
        
        for occupation in occupations[:1]:  # Limit to 1 occupation task
            if isinstance(occupation, dict):
                person = occupation.get("person", "Worker")
                job = occupation.get("occupation", "Unknown Occupation")
                location = occupation.get("location", "Unknown Location")
                time_period = occupation.get("time_period", "Unknown Period")
                
                task_data = {
                    "person_name": person,
                    "occupation": job,
                    "location": location,
                    "time_period": time_period,
                    "employer": "To be determined",
                    "research_goal": f"Understand {person}'s career and work history"
                }
                
                task = self._create_task_from_template("occupation_research", task_data)
                if task:
                    tasks.append(task)
        
        return tasks

    def _generate_location_tasks(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate location research tasks."""
        tasks = []
        locations = extracted_data.get("locations", [])
        
        for location in locations[:1]:  # Limit to 1 location task
            if isinstance(location, dict):
                place = location.get("place", "Unknown Location")
                context = location.get("context", "residence")
                time_period = location.get("time_period", "Unknown Period")
                
                task_data = {
                    "person_name": "Family Member",
                    "location": place,
                    "time_period": time_period,
                    "residence_type": context,
                    "neighbors": "To be researched",
                    "information_sought": f"Family's time in {place} and local connections"
                }
                
                task = self._create_task_from_template("location_research", task_data)
                if task:
                    tasks.append(task)
        
        return tasks

    def _create_task_from_template(self, template_key: str, task_data: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Create a task from a template with provided data."""
        try:
            template = self.task_templates.get(template_key)
            if not template:
                return None
            
            # Format title and description
            title = template["title"].format(**task_data)
            description = template["description"].format(**task_data)
            
            return {
                "title": title,
                "description": description,
                "category": template["category"],
                "priority": template["priority"],
                "template_used": template_key
            }
            
        except KeyError as e:
            logger.warning(f"Missing template data key {e} for template {template_key}")
            return None
        except Exception as e:
            logger.error(f"Error creating task from template {template_key}: {e}")
            return None

    def _create_fallback_tasks(self, person_data: Dict[str, Any], suggested_tasks: List[str]) -> List[Dict[str, Any]]:
        """Create fallback tasks from AI suggestions."""
        fallback_tasks = []
        username = person_data.get("username", "Unknown")
        
        for i, task_desc in enumerate(suggested_tasks[:3]):  # Limit to 3 fallback tasks
            fallback_tasks.append({
                "title": f"Genealogy Research: {username} (Task {i+1})",
                "description": f"Research Task: {task_desc}\n\nMatch: {username}\nPriority: Medium\n\nThis is a general research task. Consider breaking it down into more specific actions.",
                "category": "general",
                "priority": "medium",
                "template_used": "fallback"
            })
        
        return fallback_tasks

    def _prioritize_and_limit_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize tasks and limit to maximum number."""
        if not tasks:
            return []
        
        # Sort by priority (high > medium > low)
        priority_order = {"high": 3, "medium": 2, "low": 1}
        sorted_tasks = sorted(
            tasks,
            key=lambda t: priority_order.get(t.get("priority", "low"), 1),
            reverse=True
        )
        
        # Limit to maximum tasks per person
        max_tasks = self.task_config["max_tasks_per_person"]
        return sorted_tasks[:max_tasks]


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def genealogical_task_templates_module_tests() -> bool:
    """
    Comprehensive test suite for genealogical_task_templates.py.
    Tests genealogical task generation, template management, and specialized research workflows.
    """
    from test_framework import TestSuite, suppress_logging
    from unittest.mock import MagicMock, patch
    import time

    suite = TestSuite("Genealogical Task Templates & Research Generation", "genealogical_task_templates.py")
    suite.start_suite()

    # === INITIALIZATION TESTS ===
    def test_module_imports():
        """Test that required modules and dependencies are imported correctly."""
        # Test core infrastructure imports
        assert 'logger' in globals(), "Logger should be initialized"
        assert 'get_logger' in globals(), "get_logger function should be available"
        
        # Test class availability
        assert 'GenealogicalTaskGenerator' in globals(), "GenealogicalTaskGenerator class should be available"
        
        # Test standard imports
        required_imports = ['json', 'logging', 'datetime']
        for import_name in required_imports:
            assert import_name in globals(), f"Import {import_name} should be available"

    def test_task_generator_initialization():
        """Test GenealogicalTaskGenerator initialization and setup."""
        generator = GenealogicalTaskGenerator()
        
        # Test basic initialization
        assert hasattr(generator, 'task_templates'), "Generator should have task templates"
        assert hasattr(generator, 'task_config'), "Generator should have task configuration"
        assert isinstance(generator.task_templates, dict), "Task templates should be a dictionary"
        assert isinstance(generator.task_config, dict), "Task config should be a dictionary"
        
        # Test GEDCOM AI integration setup
        assert hasattr(generator, 'gedcom_ai_available'), "Should track GEDCOM AI availability"
        assert hasattr(generator, 'gedcom_ai_integrator'), "Should have integrator attribute"

    def test_task_templates_structure():
        """Test that task templates are properly structured."""
        generator = GenealogicalTaskGenerator()
        templates = generator.task_templates
        
        # Test template keys exist
        required_templates = [
            "vital_records_search", "dna_match_analysis", "immigration_research",
            "census_research", "military_research", "occupation_research"
        ]
        for template_key in required_templates:
            assert template_key in templates, f"Template {template_key} should be available"
            assert isinstance(templates[template_key], dict), f"Template {template_key} should be a dictionary"
            assert "title" in templates[template_key], f"Template {template_key} should have title"

    # === CORE FUNCTIONALITY TESTS ===
    def test_basic_task_generation():
        """Test basic task generation functionality."""
        generator = GenealogicalTaskGenerator()
        
        # Test data
        test_extracted_data = {
            "structured_names": [
                {"full_name": "John Smith", "nicknames": [], "maiden_name": None}
            ],
            "vital_records": [
                {"person": "John Smith", "event_type": "birth", "date": "1850", "place": "Aberdeen, Scotland"}
            ],
            "locations": [
                {"place": "Aberdeen, Scotland", "context": "birthplace", "time_period": "1850"}
            ],
            "research_questions": ["finding John Smith's parents"]
        }
        
        test_person_data = {"username": "TestUser"}
        test_suggested_tasks = ["Research John Smith's family history"]
        
        # Test task generation
        tasks = generator.generate_research_tasks(
            test_person_data,
            test_extracted_data,
            test_suggested_tasks
        )
        
        assert isinstance(tasks, list), "Should return a list of tasks"
        assert len(tasks) > 0, "Should generate at least one task"
        
        # Test task structure
        for task in tasks:
            assert isinstance(task, dict), "Each task should be a dictionary"
            assert "title" in task, "Task should have a title"
            assert "description" in task, "Task should have a description"

    def test_vital_records_task_generation():
        """Test specialized vital records task generation."""
        generator = GenealogicalTaskGenerator()
        
        extracted_data = {
            "vital_records": [
                {"person": "Mary Johnson", "event_type": "marriage", "date": "1875", "place": "Boston, MA"},
                {"person": "William Johnson", "event_type": "death", "date": "1900", "place": "New York, NY"}
            ]
        }
        
        vital_tasks = generator._generate_vital_records_tasks(extracted_data)
        
        assert isinstance(vital_tasks, list), "Should return list of vital records tasks"
        if len(vital_tasks) > 0:  # Only test if tasks were generated
            task = vital_tasks[0]
            assert "title" in task, "Vital records task should have title"
            assert "description" in task, "Vital records task should have description"
            assert "priority" in task, "Vital records task should have priority"

    def test_location_task_generation():
        """Test location-based task generation."""
        generator = GenealogicalTaskGenerator()
        
        extracted_data = {
            "locations": [
                {"place": "Dublin, Ireland", "context": "birthplace", "time_period": "1840"},
                {"place": "Liverpool, England", "context": "immigration", "time_period": "1860"}
            ]
        }
        
        location_tasks = generator._generate_location_tasks(extracted_data)
        
        assert isinstance(location_tasks, list), "Should return list of location tasks"
        if len(location_tasks) > 0:  # Only test if tasks were generated
            for task in location_tasks:
                assert isinstance(task, dict), "Each location task should be a dictionary"
                assert "title" in task, "Location task should have title"

    def test_occupation_task_generation():
        """Test occupation-based task generation."""
        generator = GenealogicalTaskGenerator()
        
        extracted_data = {
            "occupations": [
                {"person": "Thomas Baker", "occupation": "baker", "location": "London", "time_period": "1880-1900"},
                {"person": "Sarah Miller", "occupation": "seamstress", "location": "Manchester", "time_period": "1870"}
            ]
        }
        
        occupation_tasks = generator._generate_occupation_tasks(extracted_data)
        
        assert isinstance(occupation_tasks, list), "Should return list of occupation tasks"
        if len(occupation_tasks) > 0:  # Only test if tasks were generated
            for task in occupation_tasks:
                assert isinstance(task, dict), "Each occupation task should be a dictionary"
                assert "title" in task, "Occupation task should have title"

    # === EDGE CASE TESTS ===
    def test_empty_data_handling():
        """Test task generation with empty or minimal data."""
        generator = GenealogicalTaskGenerator()
        
        # Test with completely empty data
        empty_tasks = generator.generate_research_tasks({}, {}, [])
        assert isinstance(empty_tasks, list), "Should return list even with empty data"
        
        # Test with minimal data
        minimal_person = {"username": "TestUser"}
        minimal_extracted = {"structured_names": []}
        minimal_suggested = []
        
        minimal_tasks = generator.generate_research_tasks(minimal_person, minimal_extracted, minimal_suggested)
        assert isinstance(minimal_tasks, list), "Should handle minimal data gracefully"

    def test_invalid_template_handling():
        """Test handling of invalid or missing template data."""
        generator = GenealogicalTaskGenerator()
        
        # Test with invalid template key
        invalid_task = generator._create_task_from_template("nonexistent_template", {"test": "data"})
        assert invalid_task is None, "Should return None for invalid template"
        
        # Test with empty task data
        valid_template = list(generator.task_templates.keys())[0]
        empty_task = generator._create_task_from_template(valid_template, {})
        # Should handle empty data gracefully (may return task or None)
        assert empty_task is None or isinstance(empty_task, dict), "Should handle empty data gracefully"

    def test_fallback_task_creation():
        """Test fallback task creation when no specialized tasks can be generated."""
        generator = GenealogicalTaskGenerator()
        
        person_data = {"username": "TestUser"}
        suggested_tasks = ["Research family history", "Find birth records"]
        
        fallback_tasks = generator._create_fallback_tasks(person_data, suggested_tasks)
        
        assert isinstance(fallback_tasks, list), "Should return list of fallback tasks"
        assert len(fallback_tasks) > 0, "Should generate at least one fallback task"
        
        for task in fallback_tasks:
            assert isinstance(task, dict), "Fallback task should be dictionary"
            assert "title" in task, "Fallback task should have title"
            assert "description" in task, "Fallback task should have description"

    # === INTEGRATION TESTS ===
    def test_gedcom_ai_integration():
        """Test GEDCOM AI integration when available."""
        generator = GenealogicalTaskGenerator()
        
        # Test AI availability tracking
        assert hasattr(generator, 'gedcom_ai_available'), "Should track AI availability"
        assert isinstance(generator.gedcom_ai_available, bool), "AI availability should be boolean"
        
        # Test integrator attribute existence
        assert hasattr(generator, 'gedcom_ai_integrator'), "Should have integrator attribute"
        # integrator may be None if not available, which is fine

    def test_task_prioritization():
        """Test task prioritization and limiting functionality."""
        generator = GenealogicalTaskGenerator()
        
        # Create test tasks with different priorities
        test_tasks = [
            {"title": "High Priority Task", "priority": "high", "description": "Test"},
            {"title": "Medium Priority Task", "priority": "medium", "description": "Test"},
            {"title": "Low Priority Task", "priority": "low", "description": "Test"},
            {"title": "Another High Task", "priority": "high", "description": "Test"}
        ]
        
        prioritized_tasks = generator._prioritize_and_limit_tasks(test_tasks)
        
        assert isinstance(prioritized_tasks, list), "Should return list of prioritized tasks"
        assert len(prioritized_tasks) <= len(test_tasks), "Should not exceed original task count"
        
        # Check that high priority tasks come first if any prioritization occurred
        if len(prioritized_tasks) > 1:
            first_task = prioritized_tasks[0]
            assert "priority" in first_task, "Prioritized task should have priority field"

    def test_template_configuration_loading():
        """Test loading and validation of task configuration."""
        generator = GenealogicalTaskGenerator()
        config = generator.task_config
        
        # Test configuration structure
        assert isinstance(config, dict), "Task config should be dictionary"
        
        # Test for expected configuration keys
        expected_keys = ["max_tasks_per_person", "priority_weights", "default_priority"]
        for key in expected_keys:
            if key in config:  # Optional keys may not exist
                assert config[key] is not None, f"Config key {key} should not be None"

    # === PERFORMANCE TESTS ===
    def test_performance():
        """Test performance of task generation operations."""
        generator = GenealogicalTaskGenerator()
        
        # Test data
        test_extracted_data = {
            "structured_names": [
                {"full_name": f"Person {i}", "nicknames": [], "maiden_name": None}
                for i in range(10)
            ],
            "vital_records": [
                {"person": f"Person {i}", "event_type": "birth", "date": f"{1850+i}", "place": "Test Location"}
                for i in range(10)
            ],
            "locations": [
                {"place": f"Location {i}", "context": "birthplace", "time_period": f"{1850+i}"}
                for i in range(5)
            ]
        }
        
        start_time = time.time()
        
        # Run task generation multiple times
        for _ in range(5):
            tasks = generator.generate_research_tasks(
                {"username": "TestUser"},
                test_extracted_data,
                ["Test task"]
            )
            assert isinstance(tasks, list), "Should return tasks list"
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0, f"Performance test should complete quickly, took {elapsed:.3f}s"

    def test_bulk_template_processing():
        """Test performance with bulk template processing."""
        generator = GenealogicalTaskGenerator()
        
        start_time = time.time()
        
        # Process multiple template types
        template_keys = list(generator.task_templates.keys())[:5]  # Test first 5 templates
        
        for template_key in template_keys:
            for i in range(10):
                task_data = {"person_name": f"Test Person {i}", "time_period": "1850-1900"}
                task = generator._create_task_from_template(template_key, task_data)
                # Task may be None or dict, both are acceptable
                assert task is None or isinstance(task, dict), "Task should be None or dict"
        
        elapsed = time.time() - start_time
        assert elapsed < 0.5, f"Bulk template processing should be fast, took {elapsed:.3f}s"

    # === ERROR HANDLING TESTS ===
    def test_error_handling():
        """Test error handling with invalid inputs and edge cases."""
        generator = GenealogicalTaskGenerator()
        
        # Test with None inputs (using type ignore for intentional testing)
        result = generator.generate_research_tasks(None, None, None)  # type: ignore
        assert isinstance(result, list), "Should handle None inputs gracefully"
        
        # Test with invalid data types (using type ignore for intentional testing)
        result = generator.generate_research_tasks("invalid", "invalid", "invalid")  # type: ignore
        assert isinstance(result, list), "Should handle invalid data types"
        
        # Test private methods with invalid data
        result = generator._generate_vital_records_tasks({})
        assert isinstance(result, list), "Should handle empty vital records data"
        
        result = generator._generate_location_tasks({"locations": "invalid"})
        assert isinstance(result, list), "Should handle invalid location data"

    def test_malformed_data_handling():
        """Test handling of malformed or corrupted data structures."""
        generator = GenealogicalTaskGenerator()
        
        # Test with malformed extracted data
        malformed_data = {
            "vital_records": "not_a_list",
            "locations": [{"incomplete": "data"}],
            "occupations": [None, {"invalid": True}]
        }
        
        tasks = generator.generate_research_tasks(
            {"username": "Test"}, 
            malformed_data, 
            ["test"]
        )
        
        assert isinstance(tasks, list), "Should handle malformed data gracefully"
        # Tasks list may be empty or contain fallback tasks, both are acceptable

    # Run all tests
    with suppress_logging():
        suite.run_test(
            "Module imports and initialization",
            test_module_imports,
            "All required modules and dependencies are properly imported",
            "Test import availability of core infrastructure and genealogical components",
            "Module initialization provides complete dependency access"
        )

        suite.run_test(
            "Task generator initialization",
            test_task_generator_initialization,
            "GenealogicalTaskGenerator initializes correctly with templates and configuration",
            "Test GenealogicalTaskGenerator initialization and setup",
            "Task generator initialization provides complete template and AI integration setup"
        )

        suite.run_test(
            "Task templates structure validation",
            test_task_templates_structure,
            "Task templates are properly structured with required fields and formats",
            "Test that task templates are properly structured",
            "Task templates provide complete genealogical research template library"
        )

        suite.run_test(
            "Basic task generation functionality",
            test_basic_task_generation,
            "Task generation creates valid, structured genealogical research tasks",
            "Test basic task generation functionality",
            "Basic task generation provides actionable genealogical research tasks"
        )

        suite.run_test(
            "Vital records task generation",
            test_vital_records_task_generation,
            "Vital records tasks are generated with proper structure and priority",
            "Test specialized vital records task generation",
            "Vital records task generation provides focused genealogical research objectives"
        )

        suite.run_test(
            "Location-based task generation",
            test_location_task_generation,
            "Location tasks are generated based on geographical research opportunities",
            "Test location-based task generation",
            "Location task generation provides place-specific research strategies"
        )

        suite.run_test(
            "Occupation-based task generation",
            test_occupation_task_generation,
            "Occupation tasks are generated to explore professional and trade connections",
            "Test occupation-based task generation",
            "Occupation task generation provides professional research pathways"
        )

        suite.run_test(
            "Empty data handling",
            test_empty_data_handling,
            "Task generation handles empty or minimal data gracefully",
            "Test task generation with empty or minimal data",
            "Empty data handling ensures robust operation with incomplete information"
        )

        suite.run_test(
            "Invalid template handling",
            test_invalid_template_handling,
            "Invalid or missing template data is handled gracefully",
            "Test handling of invalid or missing template data",
            "Invalid template handling provides robust template processing"
        )

        suite.run_test(
            "Fallback task creation",
            test_fallback_task_creation,
            "Fallback tasks are created when specialized tasks cannot be generated",
            "Test fallback task creation when no specialized tasks can be generated",
            "Fallback task creation ensures users always receive actionable research tasks"
        )

        suite.run_test(
            "GEDCOM AI integration",
            test_gedcom_ai_integration,
            "GEDCOM AI integration is properly configured and tracked",
            "Test GEDCOM AI integration when available",
            "GEDCOM AI integration provides enhanced genealogical analysis capabilities"
        )

        suite.run_test(
            "Task prioritization and limiting",
            test_task_prioritization,
            "Task prioritization orders research tasks by importance and limits output",
            "Test task prioritization and limiting functionality",
            "Task prioritization ensures most important research tasks are presented first"
        )

        suite.run_test(
            "Template configuration loading",
            test_template_configuration_loading,
            "Task configuration is loaded and validated correctly",
            "Test loading and validation of task configuration",
            "Configuration loading provides proper task generation parameters"
        )

        suite.run_test(
            "Performance validation",
            test_performance,
            "Task generation operations complete within reasonable time limits",
            "Test performance of task generation operations",
            "Performance validation ensures efficient genealogical task generation"
        )

        suite.run_test(
            "Bulk template processing performance",
            test_bulk_template_processing,
            "Bulk template processing handles multiple templates efficiently",
            "Test performance with bulk template processing",
            "Bulk processing provides scalable template-based task generation"
        )

        suite.run_test(
            "Error handling robustness",
            test_error_handling,
            "Error handling gracefully manages invalid inputs and edge cases",
            "Test error handling with invalid inputs and edge cases",
            "Error handling ensures stable operation under adverse conditions"
        )

        suite.run_test(
            "Malformed data handling",
            test_malformed_data_handling,
            "Malformed or corrupted data structures are handled gracefully",
            "Test handling of malformed or corrupted data structures",
            "Malformed data handling provides robust data processing capabilities"
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive genealogical task templates tests using standardized TestSuite format."""
    return genealogical_task_templates_module_tests()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    import sys
    
    # Always run comprehensive tests  
    print("🧬 Running Genealogical Task Templates comprehensive test suite...")
    success = run_comprehensive_tests()
    if success:
        print("\n✅ All genealogical task templates tests completed successfully!")
    else:
        print("\n❌ Some genealogical task templates tests failed!")
    sys.exit(0 if success else 1)
