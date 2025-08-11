"""
Enhanced Message Personalization System for Ancestry Project

This module provides dynamic message generation capabilities that leverage
extracted genealogical data to create personalized, engaging messages.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 9.1 - Message Template Enhancement
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)


class MessagePersonalizer:
    """
    Enhanced message personalization system that creates dynamic, 
    genealogically-informed messages based on extracted data.
    """

    def __init__(self):
        """Initialize the message personalizer with templates and configuration."""
        self.templates = self._load_message_templates()
        self.personalization_config = self._load_personalization_config()

    def _load_message_templates(self) -> Dict[str, str]:
        """Load message templates from messages.json."""
        try:
            script_dir = Path(__file__).resolve().parent
            messages_path = script_dir / "messages.json"
            
            if not messages_path.exists():
                logger.error(f"messages.json not found at {messages_path}")
                return {}
            
            with messages_path.open("r", encoding="utf-8") as f:
                templates = json.load(f)
            
            logger.debug(f"Loaded {len(templates)} message templates")
            return templates
            
        except Exception as e:
            logger.error(f"Error loading message templates: {e}")
            return {}

    def _load_personalization_config(self) -> Dict[str, Any]:
        """Load personalization configuration settings."""
        return {
            "max_ancestors_to_mention": 3,
            "max_locations_to_mention": 2,
            "max_research_questions": 2,
            "include_dates_in_context": True,
            "include_occupations": True,
            "geographic_context_priority": ["Scotland", "Ireland", "England", "Poland", "Ukraine"]
        }

    def create_personalized_message(
        self,
        template_key: str,
        person_data: Dict[str, Any],
        extracted_data: Dict[str, Any],
        base_format_data: Dict[str, str]
    ) -> str:
        """
        Create a personalized message using extracted genealogical data.
        
        Args:
            template_key: Key for the message template to use
            person_data: Information about the person being messaged
            extracted_data: Genealogical data extracted from conversations
            base_format_data: Basic formatting data (name, relationships, etc.)
            
        Returns:
            Personalized message text
        """
        try:
            # Get the template
            if template_key not in self.templates:
                logger.warning(f"Template '{template_key}' not found, using fallback")
                template_key = self._get_fallback_template(template_key)
            
            template = self.templates.get(template_key, "")
            if not template:
                return self._create_fallback_message(person_data, base_format_data)
            
            # Create enhanced format data
            enhanced_format_data = self._create_enhanced_format_data(
                extracted_data, base_format_data, person_data
            )
            
            # Format the message with safe formatting
            try:
                personalized_message = template.format(**enhanced_format_data)
            except KeyError as ke:
                logger.warning(f"Missing template key {ke}, using fallback formatting")
                # Add missing keys with default values
                missing_keys = {
                    "total_rows": "many",
                    "predicted_relationship": "family connection",
                    "actual_relationship": "family connection",
                    "relationship_path": "our shared family line",
                    "shared_ancestors": "our shared family line",
                    "ancestors_details": "",
                    "genealogical_context": "I'm excited to learn more about our family connection.",
                    "research_focus": " our shared family history",
                    "specific_questions": "Do you have any additional family information that might help our research?",
                    "geographic_context": "Family History Research",
                    "location_context": "",
                    "research_suggestions": "I'd love to learn more about your family history.",
                    "specific_research_questions": "",
                    "mentioned_people": "your family history",
                    "research_context": "our shared family history",
                    "personalized_response": "I found your information very helpful for my genealogical research.",
                    "research_insights": "This information is very valuable for our family research.",
                    "follow_up_questions": "Do you have any other family documents or stories that might help our research?",
                    "estimated_relationship": "close family connection",
                    "shared_dna_amount": "significant DNA",
                    "dna_context": "This suggests we share recent common ancestors.",
                    "shared_ancestor_information": "I'd love to compare our family trees to identify our common ancestors.",
                    "research_collaboration_request": "Would you be interested in collaborating on our genealogical research?",
                    "research_topic": "Family History Research",
                    "specific_research_needs": "I'm looking for additional information to complete our family history.",
                    "collaboration_proposal": "Perhaps we could share our research findings and work together to solve any family history mysteries."
                }

                # Add any missing keys to enhanced_format_data
                for key, default_value in missing_keys.items():
                    if key not in enhanced_format_data:
                        enhanced_format_data[key] = default_value

                # Try formatting again
                personalized_message = template.format(**enhanced_format_data)
            
            logger.info(f"Created personalized message using template '{template_key}'")
            return personalized_message
            
        except Exception as e:
            logger.error(f"Error creating personalized message: {e}")
            return self._create_fallback_message(person_data, base_format_data)

    def _create_enhanced_format_data(
        self,
        extracted_data: Dict[str, Any],
        base_format_data: Dict[str, str],
        person_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create enhanced formatting data from extracted genealogical information."""
        enhanced_data = base_format_data.copy()
        
        # Add genealogical context
        enhanced_data.update({
            "shared_ancestors": self._format_shared_ancestors(extracted_data),
            "ancestors_details": self._format_ancestor_details(extracted_data),
            "genealogical_context": self._create_genealogical_context(extracted_data),
            "research_focus": self._identify_research_focus(extracted_data),
            "specific_questions": self._generate_specific_questions(extracted_data),
            "geographic_context": self._create_geographic_context(extracted_data),
            "location_context": self._format_location_context(extracted_data),
            "research_suggestions": self._create_research_suggestions(extracted_data),
            "specific_research_questions": self._format_research_questions(extracted_data),
            "mentioned_people": self._format_mentioned_people(extracted_data),
            "research_context": self._create_research_context(extracted_data),
            "personalized_response": self._create_personalized_response(extracted_data),
            "research_insights": self._create_research_insights(extracted_data),
            "follow_up_questions": self._create_follow_up_questions(extracted_data),
            "estimated_relationship": self._format_estimated_relationship(extracted_data),
            "shared_dna_amount": self._format_shared_dna(extracted_data),
            "dna_context": self._create_dna_context(extracted_data),
            "shared_ancestor_information": self._format_shared_ancestor_info(extracted_data),
            "research_collaboration_request": self._create_collaboration_request(extracted_data),
            "research_topic": self._identify_research_topic(extracted_data),
            "specific_research_needs": self._format_research_needs(extracted_data),
            "collaboration_proposal": self._create_collaboration_proposal(extracted_data)
        })
        
        return enhanced_data

    def _format_shared_ancestors(self, extracted_data: Dict[str, Any]) -> str:
        """Format shared ancestors information."""
        structured_names = extracted_data.get("structured_names", [])
        if not structured_names:
            return "our shared family line"
        
        # Get up to 3 most relevant names
        ancestor_names = []
        for name_data in structured_names[:self.personalization_config["max_ancestors_to_mention"]]:
            if isinstance(name_data, dict):
                full_name = name_data.get("full_name", "")
                if full_name:
                    ancestor_names.append(full_name)
            elif isinstance(name_data, str):
                ancestor_names.append(name_data)
        
        if not ancestor_names:
            return "our shared family line"
        elif len(ancestor_names) == 1:
            return ancestor_names[0]
        elif len(ancestor_names) == 2:
            return f"{ancestor_names[0]} and {ancestor_names[1]}"
        else:
            return f"{', '.join(ancestor_names[:-1])}, and {ancestor_names[-1]}"

    def _format_ancestor_details(self, extracted_data: Dict[str, Any]) -> str:
        """Format detailed ancestor information."""
        vital_records = extracted_data.get("vital_records", [])
        if not vital_records:
            return ""
        
        details = []
        for record in vital_records[:2]:  # Limit to 2 most relevant records
            if isinstance(record, dict):
                person = record.get("person", "")
                event_type = record.get("event_type", "")
                date = record.get("date", "")
                place = record.get("place", "")
                
                if person and (date or place):
                    detail_parts = [person]
                    if event_type and date:
                        detail_parts.append(f"{event_type} {date}")
                    if place:
                        detail_parts.append(f"in {place}")
                    details.append(" ".join(detail_parts))
        
        if details:
            return f" ({'; '.join(details)})"
        return ""

    def _create_genealogical_context(self, extracted_data: Dict[str, Any]) -> str:
        """Create genealogical context paragraph."""
        context_parts = []
        
        # Add location information
        locations = extracted_data.get("locations", [])
        if locations:
            location_text = self._format_location_context(extracted_data)
            if location_text:
                context_parts.append(f"Our family history traces through{location_text}.")
        
        # Add occupation information
        occupations = extracted_data.get("occupations", [])
        if occupations and self.personalization_config["include_occupations"]:
            occ_text = self._format_occupations(occupations)
            if occ_text:
                context_parts.append(occ_text)
        
        return " ".join(context_parts) if context_parts else "I'm excited to learn more about our family connection."

    def _format_occupations(self, occupations: List[Any]) -> str:
        """Format occupation information."""
        if not occupations:
            return ""
        
        occ_descriptions = []
        for occ in occupations[:2]:  # Limit to 2 occupations
            if isinstance(occ, dict):
                person = occ.get("person", "")
                occupation = occ.get("occupation", "")
                if person and occupation:
                    occ_descriptions.append(f"{person} worked as a {occupation}")
        
        if occ_descriptions:
            return f"Family records show {', and '.join(occ_descriptions)}."
        return ""

    def _identify_research_focus(self, extracted_data: Dict[str, Any]) -> str:
        """Identify the main research focus from extracted data."""
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            # Use the first research question as focus
            return f" {research_questions[0].lower()}"
        
        # Fallback to general family history
        return " our shared family history"

    def _generate_specific_questions(self, extracted_data: Dict[str, Any]) -> str:
        """Generate specific follow-up questions based on extracted data."""
        questions = []
        
        # Questions based on research gaps
        research_questions = extracted_data.get("research_questions", [])
        for question in research_questions[:self.personalization_config["max_research_questions"]]:
            if isinstance(question, str) and question:
                questions.append(f"Do you have any information about {question.lower()}?")
        
        # Questions based on locations
        locations = extracted_data.get("locations", [])
        for location in locations[:1]:  # Just one location question
            if isinstance(location, dict):
                place = location.get("place", "")
                if place:
                    questions.append(f"Do you have any family connections to {place}?")
        
        if questions:
            return " ".join(questions)
        return "Do you have any additional family information that might help our research?"

    def _create_geographic_context(self, extracted_data: Dict[str, Any]) -> str:
        """Create geographic context for the subject line."""
        locations = extracted_data.get("locations", [])
        if not locations:
            return "Family History Research"
        
        # Get the most relevant location
        for location in locations:
            if isinstance(location, dict):
                place = location.get("place", "")
                if place:
                    return place
        
        return "Family History Research"

    def _format_location_context(self, extracted_data: Dict[str, Any]) -> str:
        """Format location context for messages."""
        locations = extracted_data.get("locations", [])
        if not locations:
            return ""
        
        location_names = []
        for location in locations[:self.personalization_config["max_locations_to_mention"]]:
            if isinstance(location, dict):
                place = location.get("place", "")
                if place:
                    location_names.append(place)
        
        if location_names:
            if len(location_names) == 1:
                return f" {location_names[0]}"
            else:
                return f" {', '.join(location_names[:-1])}, and {location_names[-1]}"
        return ""

    def _create_research_suggestions(self, extracted_data: Dict[str, Any]) -> str:
        """Create research suggestions based on extracted data."""
        suggestions = []
        
        # Suggestions based on mentioned people
        structured_names = extracted_data.get("structured_names", [])
        if structured_names:
            name = structured_names[0].get("full_name", "") if isinstance(structured_names[0], dict) else str(structured_names[0])
            if name:
                suggestions.append(f"I'm particularly interested in learning more about {name} and their family line.")
        
        # Suggestions based on research questions
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            suggestions.append(f"I'm currently researching {research_questions[0].lower()}.")
        
        return " ".join(suggestions) if suggestions else "I'd love to learn more about your family history."

    def _format_research_questions(self, extracted_data: Dict[str, Any]) -> str:
        """Format specific research questions."""
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            return f" - particularly about {research_questions[0].lower()}"
        return ""

    def _get_fallback_template(self, original_key: str) -> str:
        """Get a fallback template key based on the original key."""
        if "In_Tree" in original_key:
            return "In_Tree-Initial"
        elif "Out_Tree" in original_key:
            return "Out_Tree-Initial"
        elif "Productive" in original_key:
            return "Productive_Reply_Acknowledgement"
        else:
            return "In_Tree-Initial"  # Default fallback

    def _create_fallback_message(self, person_data: Dict[str, Any], base_format_data: Dict[str, str]) -> str:
        """Create a simple fallback message when template processing fails."""
        name = base_format_data.get("name", "there")
        return f"Dear {name},\n\nThank you for connecting! I'm excited to learn more about our family history.\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland"

    # Additional helper methods for remaining format data fields
    def _format_mentioned_people(self, extracted_data: Dict[str, Any]) -> str:
        """Format mentioned people from extracted data."""
        structured_names = extracted_data.get("structured_names", [])
        if not structured_names:
            return "your family history"
        
        names = []
        for name_data in structured_names[:3]:
            if isinstance(name_data, dict):
                full_name = name_data.get("full_name", "")
                if full_name:
                    names.append(full_name)
        
        if names:
            if len(names) == 1:
                return names[0]
            elif len(names) == 2:
                return f"{names[0]} and {names[1]}"
            else:
                return f"{', '.join(names[:-1])}, and {names[-1]}"
        return "your family history"

    def _create_research_context(self, extracted_data: Dict[str, Any]) -> str:
        """Create research context description."""
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            return research_questions[0]
        
        # Fallback to general context
        locations = extracted_data.get("locations", [])
        if locations and isinstance(locations[0], dict):
            place = locations[0].get("place", "")
            if place:
                return f"family connections in {place}"
        
        return "our shared family history"

    def _create_personalized_response(self, extracted_data: Dict[str, Any]) -> str:
        """Create personalized response content."""
        # This would be populated by AI-generated content
        return "I found your information very helpful for my genealogical research."

    def _create_research_insights(self, extracted_data: Dict[str, Any]) -> str:
        """Create research insights based on extracted data."""
        insights = []
        
        # Insights from vital records
        vital_records = extracted_data.get("vital_records", [])
        if vital_records:
            insights.append("This information helps fill in some important gaps in our family timeline.")
        
        # Insights from relationships
        relationships = extracted_data.get("relationships", [])
        if relationships:
            insights.append("The family relationships you mentioned help clarify some connections I've been researching.")
        
        return " ".join(insights) if insights else "This information is very valuable for our family research."

    def _create_follow_up_questions(self, extracted_data: Dict[str, Any]) -> str:
        """Create follow-up questions for continued research."""
        questions = []
        
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            questions.append(f"Do you have any additional information about {research_questions[0].lower()}?")
        
        if not questions:
            questions.append("Do you have any other family documents or stories that might help our research?")
        
        return " ".join(questions)

    def _format_estimated_relationship(self, extracted_data: Dict[str, Any]) -> str:
        """Format estimated relationship from DNA data."""
        dna_info = extracted_data.get("dna_information", [])
        for info in dna_info:
            if "cousin" in str(info).lower() or "relationship" in str(info).lower():
                return str(info)
        return "close family connection"

    def _format_shared_dna(self, extracted_data: Dict[str, Any]) -> str:
        """Format shared DNA amount."""
        dna_info = extracted_data.get("dna_information", [])
        for info in dna_info:
            if "cm" in str(info).lower() or "centimorgans" in str(info).lower():
                return str(info)
        return "significant DNA"

    def _create_dna_context(self, extracted_data: Dict[str, Any]) -> str:
        """Create DNA-specific context."""
        return "This suggests we share recent common ancestors."

    def _format_shared_ancestor_info(self, extracted_data: Dict[str, Any]) -> str:
        """Format shared ancestor information."""
        return "I'd love to compare our family trees to identify our common ancestors."

    def _create_collaboration_request(self, extracted_data: Dict[str, Any]) -> str:
        """Create collaboration request text."""
        return "Would you be interested in collaborating on our genealogical research?"

    def _identify_research_topic(self, extracted_data: Dict[str, Any]) -> str:
        """Identify the main research topic."""
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            return research_questions[0]
        return "Family History Research"

    def _format_research_needs(self, extracted_data: Dict[str, Any]) -> str:
        """Format specific research needs."""
        return "I'm looking for additional information to complete our family history."

    def _create_collaboration_proposal(self, extracted_data: Dict[str, Any]) -> str:
        """Create collaboration proposal text."""
        return "Perhaps we could share our research findings and work together to solve any family history mysteries."


# Test functions
def test_message_personalization():
    """Test the message personalization system."""
    logger.info("Testing message personalization system...")
    
    personalizer = MessagePersonalizer()
    
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
    test_base_data = {
        "name": "TestUser",
        "predicted_relationship": "3rd cousin",
        "actual_relationship": "3rd cousin",
        "relationship_path": "Through John Smith (1850-1920)",
        "total_rows": "150"
    }
    
    # Test message creation
    message = personalizer.create_personalized_message(
        "Enhanced_In_Tree-Initial",
        test_person_data,
        test_extracted_data,
        test_base_data
    )
    
    success = len(message) > 100 and "John Smith" in message
    logger.info(f"Message personalization test: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_fallback_template_path():
    """Ensure an unknown template key triggers safe fallback without exception."""
    personalizer = MessagePersonalizer()
    # Force empty templates to guarantee fallback path
    personalizer.templates = {"In_Tree-Initial": "Hello {name}!"}
    msg = personalizer.create_personalized_message(
        "Totally_Unknown_Template",
        {"username": "UserX"},
        {},
        {"name": "UserX"}
    )
    # Either fallback message or resolved fallback template must appear
    return ("UserX" in msg) and len(msg) > 10

def test_shared_ancestors_formatting():
    """Validate proper Oxford-comma style formatting for multiple ancestors."""
    p = MessagePersonalizer()
    data = {"structured_names": [
        {"full_name": "Alice Brown"},
        {"full_name": "Robert Clark"},
        {"full_name": "Sarah Davis"},
    ]}
    formatted = p._format_shared_ancestors(data)
    # Expect: "Alice Brown, Robert Clark, and Sarah Davis"
    return formatted.count(",") == 2 and formatted.endswith("Sarah Davis") and " and " in formatted

def test_location_context_limit():
    """Ensure location context respects max_locations_to_mention constraint."""
    p = MessagePersonalizer()
    p.personalization_config["max_locations_to_mention"] = 2
    data = {"locations": [
        {"place": "Aberdeen"},
        {"place": "Glasgow"},
        {"place": "Edinburgh"},
    ]}
    ctx = p._format_location_context(data)
    # Should include only two locations and the word 'and' between them
    # Pattern: ' Aberdeen and Glasgow' (order preserved, third excluded)
    return ctx.strip().count(" ") <= 2 and "Edinburgh" not in ctx and "and" in ctx


def message_personalization_module_tests() -> bool:
    """
    Comprehensive test suite for message_personalization.py with real functionality testing.
    Tests message personalization, template generation, and intelligent messaging systems.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Message Personalization & Template Generation", "message_personalization.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Message personalization system",
            test_message_personalization,
            "Complete message personalization with template generation and intelligent messaging",
            "Test message personalization system with real template processing",
            "Test MessagePersonalizer with sample DNA match data and personalized message generation",
        )
        suite.run_test(
            "Fallback template path",
            test_fallback_template_path,
            "Fallback handling when template key missing",
            "Unknown template key -> safe fallback",
            "Ensures graceful degradation without exceptions",
        )
        suite.run_test(
            "Shared ancestors formatting",
            test_shared_ancestors_formatting,
            "Comma & conjunction formatting for multiple ancestors",
            "Formatting correctness for 3 names",
            "Prevents awkward grammar in personalized output",
        )
        suite.run_test(
            "Location context limiting",
            test_location_context_limit,
            "Respect max_locations_to_mention setting",
            "Avoids overlong geographic context strings",
            "Guarantees UX brevity rule for location mentions",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive message personalization tests using standardized TestSuite format."""
    return message_personalization_module_tests()


if __name__ == "__main__":
    """
    Execute comprehensive message personalization tests when run directly.
    Tests message personalization, template generation, and intelligent messaging systems.
    """
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
