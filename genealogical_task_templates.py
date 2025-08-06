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
        suggested_tasks: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate specialized research tasks based on extracted genealogical data.
        
        Args:
            person_data: Information about the person being researched
            extracted_data: Genealogical data extracted from conversations
            suggested_tasks: Basic AI-generated task suggestions
            
        Returns:
            List of enhanced task dictionaries with titles, descriptions, categories, and priorities
        """
        try:
            enhanced_tasks = []
            
            # Generate tasks based on extracted data types
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
            
            logger.info(f"Generated {len(prioritized_tasks)} enhanced research tasks")
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


# Test functions
def test_genealogical_task_generation():
    """Test the genealogical task generation system."""
    logger.info("Testing genealogical task generation system...")
    
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
        "occupations": [
            {"person": "John Smith", "occupation": "fisherman", "location": "Aberdeen", "time_period": "1870-1900"}
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
    
    success = len(tasks) > 0 and all("title" in task and "description" in task for task in tasks)
    logger.info(f"Task generation test: {'✅ PASSED' if success else '❌ FAILED'}")
    logger.info(f"Generated {len(tasks)} tasks")
    
    return success


if __name__ == "__main__":
    """Test suite for genealogical_task_templates.py"""
    test_genealogical_task_generation()
