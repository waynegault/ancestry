"""Message Personalization System.

Generates personalized, context-aware messages for genealogical research
collaboration by leveraging extracted family tree data, DNA information,
and research context.

Key Features:
- AI-powered message personalization
- Template-based message generation
- Context extraction from genealogical data
- Effectiveness tracking and analytics
- Multi-template support (initial contact, follow-ups, etc.)

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 9.1 - Message Template Enhancement
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)


class MessagePersonalizer:
    """
    Enhanced message personalization system that creates dynamic,
    genealogically-informed messages based on extracted data.
    """

    def __init__(self) -> None:
        """Initialize the message personalizer with templates and configuration."""
        self.templates = self._load_message_templates()
        self.personalization_config = self._load_personalization_config()
        self.effectiveness_tracker = MessageEffectivenessTracker()
        self.ab_testing_enabled = True
        self.personalization_functions_registry = self._build_personalization_registry()
        self._last_generation_snapshot: Optional[dict[str, Any]] = None

    @staticmethod
    def _load_message_templates() -> dict[str, str]:
        """Load message templates from database MessageTemplate table."""
        try:
            from database import MessageTemplate
            from session_utils import get_global_session

            session_manager = get_global_session()
            if not session_manager:
                logger.error("Global session not available. main.py must register it before loading templates.")
                return {}

            with session_manager.get_db_conn_context() as session:
                if not session:
                    logger.error("Could not get database session for template loading")
                    return {}

                # Fetch all templates from database
                templates_query = session.query(MessageTemplate).all()

                # Build dictionary with full message content (subject + body)
                templates = {}
                for template in templates_query:
                    # Reconstruct full message content with subject line
                    if template.subject_line and template.message_content:
                        full_content = f"Subject: {template.subject_line}\n\n{template.message_content}"
                    elif template.message_content:
                        full_content = template.message_content
                    else:
                        continue

                    templates[template.template_key] = full_content

                logger.debug(f"Loaded {len(templates)} message templates from database")
                return templates

        except Exception as e:
            logger.error(f"Error loading message templates from database: {e}")
            return {}

    @staticmethod
    def _load_personalization_config() -> dict[str, Any]:
        """Load personalization configuration settings."""
        return {
            "max_ancestors_to_mention": 3,
            "max_locations_to_mention": 2,
            "max_research_questions": 2,
            "include_dates_in_context": True,
            "include_occupations": True,
            "geographic_context_priority": ["Scotland", "Ireland", "England", "Poland", "Ukraine"],
            "ab_testing_split_ratio": 0.5,  # 50/50 split for A/B testing
            "min_usage_for_optimization": 10,  # Minimum usage before optimization kicks in
            "effectiveness_threshold": 6.0  # Minimum effectiveness score to consider template good
        }

    def _build_personalization_registry(self) -> dict[str, Callable[[dict[str, Any]], str]]:
        """Build registry of all personalization functions for dynamic usage."""
        return {
            # Existing functions
            "shared_ancestors": self._format_shared_ancestors,
            "ancestor_details": self._format_ancestor_details,
            "genealogical_context": self._create_genealogical_context,
            "research_focus": self._identify_research_focus,
            "specific_questions": self._generate_specific_questions,
            "geographic_context": self._create_geographic_context,
            "location_context": self._format_location_context,
            "research_suggestions": self._create_research_suggestions,
            "specific_research_questions": self._format_research_questions,
            "mentioned_people": self._format_mentioned_people,
            "research_context": self._create_research_context,
            "personalized_response": self._create_personalized_response,
            "research_insights": self._create_research_insights,
            "follow_up_questions": self._create_follow_up_questions,
            "estimated_relationship": self._format_estimated_relationship,
            "shared_dna_amount": self._format_shared_dna,
            "dna_context": self._create_dna_context,
            "shared_ancestor_information": self._format_shared_ancestor_info,
            "research_collaboration_request": self._create_collaboration_request,
            "research_topic": self._identify_research_topic,
            "specific_research_needs": self._format_research_needs,
            "collaboration_proposal": self._create_collaboration_proposal,

            # New advanced functions
            "dna_segment_analysis": self._create_dna_segment_analysis,
            "migration_pattern_context": self._create_migration_pattern_context,
            "historical_context_analysis": self._create_historical_context_analysis,
            "record_availability_assessment": self._create_record_availability_assessment,
            "dna_ethnicity_correlation": self._create_dna_ethnicity_correlation,
            "surname_distribution_analysis": self._create_surname_distribution_analysis,
            "occupation_social_context": self._create_occupation_social_context,
            "family_size_analysis": self._create_family_size_analysis,
            "generational_gap_analysis": self._create_generational_gap_analysis,
            "document_preservation_likelihood": self._create_document_preservation_likelihood
        }

    def create_personalized_message(
        self,
        template_key: str,
        person_data: Optional[dict[str, Any]] = None,
        extracted_data: Optional[dict[str, Any]] = None,
        base_format_data: Optional[dict[str, str]] = None,
        track_effectiveness: bool = True,
    ) -> tuple[str, list[str]]:
        """
        Create a personalized message using extracted genealogical data with intelligent function selection.

        Args:
            template_key: Key for the message template to use
            person_data: Information about the person being messaged
            extracted_data: Genealogical data extracted from conversations
            base_format_data: Basic formatting data (name, relationships, etc.)
            track_effectiveness: Whether to track effectiveness for optimization

        Returns:
            Tuple of (personalized message text, list of personalization functions used)
        """
        person_payload: dict[str, Any] = person_data or {}
        extraction_payload: dict[str, Any] = extracted_data or {}
        normalized_base_format = self._prepare_base_format_data(person_payload, base_format_data)

        try:
            resolved_key, template = self._resolve_template(template_key, extraction_payload)
            if not template:
                return self._create_fallback_message(person_payload, normalized_base_format), []

            selected_functions = self._select_optimal_personalization_functions(extraction_payload)
            enhanced_format_data = self._create_enhanced_format_data(
                extraction_payload, normalized_base_format, person_payload, selected_functions
            )
            personalized_message = self._format_template_safe(template, enhanced_format_data)

            if track_effectiveness:
                self._record_generation_snapshot(resolved_key, selected_functions, extraction_payload)
            else:
                logger.debug("Effectiveness tracking disabled for template '%s'", resolved_key)

            logger.info(
                "Created personalized message using template '%s' with %d personalization functions",
                resolved_key,
                len(selected_functions),
            )
            return personalized_message, selected_functions

        except Exception as e:
            logger.error(f"Error creating personalized message: {e}")
            return self._create_fallback_message(person_payload, normalized_base_format), []

    @staticmethod
    def _prepare_base_format_data(
        person_payload: dict[str, Any],
        base_format_data: Optional[dict[str, str]],
    ) -> dict[str, str]:
        """Normalize and populate base format data with safe defaults."""
        normalized_base_format = {key: str(value) for key, value in (base_format_data or {}).items()}
        if normalized_base_format:
            return normalized_base_format

        fallback_name = (
            person_payload.get("name")
            or person_payload.get("username")
            or person_payload.get("preferred_name")
            or "there"
        )
        return {"name": str(fallback_name)}

    def _resolve_template(
        self,
        template_key: str,
        extraction_payload: dict[str, Any],
    ) -> tuple[str, str]:
        """Resolve final template key (with fallbacks and A/B testing) and content."""
        resolved_key = template_key
        if resolved_key not in self.templates:
            logger.warning("Template '%s' not found, using fallback", resolved_key)
            resolved_key = self._get_fallback_template(resolved_key)

        if self.ab_testing_enabled:
            resolved_key = self._apply_ab_testing(resolved_key, extraction_payload)

        template = self.templates.get(resolved_key, "")
        return resolved_key, template

    def _format_template_safe(self, template: str, enhanced_format_data: dict[str, str]) -> str:
        """Format template text while backfilling missing keys."""
        try:
            return template.format(**enhanced_format_data)
        except KeyError as ke:
            logger.warning("Missing template key %s, using fallback formatting", ke)
            populated = enhanced_format_data.copy()
            for key, default_value in self._get_template_default_values().items():
                populated.setdefault(key, default_value)
            return template.format(**populated)

    @staticmethod
    def _get_template_default_values() -> dict[str, str]:
        """Return fallback template values for backwards compatibility."""
        return {
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
            "collaboration_proposal": "Perhaps we could share our research findings and work together to solve any family history mysteries.",
        }

    def _record_generation_snapshot(
        self,
        template_key: str,
        selected_functions: list[str],
        extraction_payload: dict[str, Any],
    ) -> None:
        """Capture telemetry for the generated message."""
        self._last_generation_snapshot = {
            "template_key": template_key,
            "selected_functions": list(selected_functions),
            "data_keys": sorted(extraction_payload.keys()),
            "timestamp": datetime.now().isoformat(),
        }

    def _apply_ab_testing(self, template_key: str, extracted_data: Optional[dict[str, Any]] = None) -> str:
        """Apply A/B testing for template selection based on effectiveness data."""
        # Get alternative templates for A/B testing
        alternative_templates = self._get_alternative_templates(template_key)

        if not alternative_templates:
            return template_key

        data_profile = extracted_data or {}
        if data_profile.get("dna_information"):
            dna_ready_templates = [tpl for tpl in alternative_templates if "dna" in tpl.lower()]
            if dna_ready_templates:
                logger.debug(
                    "A/B testing: selected DNA-focused template '%s' based on extracted data",
                    dna_ready_templates[0],
                )
                template_key = dna_ready_templates[0]

        # Check effectiveness scores
        current_score = self.effectiveness_tracker.get_template_effectiveness_score(template_key)

        for alt_template in alternative_templates:
            alt_score = self.effectiveness_tracker.get_template_effectiveness_score(alt_template)

            # If alternative is significantly better, use it
            if alt_score > current_score + 1.0:
                logger.info(f"A/B testing: switching from '{template_key}' to '{alt_template}' (score: {alt_score:.1f} vs {current_score:.1f})")
                return alt_template

        return template_key

    def _get_alternative_templates(self, template_key: str) -> list[str]:
        """Get alternative templates for A/B testing."""
        # Define template families for A/B testing
        template_families = {
            "initial_contact": ["initial_contact_v2", "initial_contact_research_focused"],
            "follow_up": ["follow_up_detailed", "follow_up_casual"],
            "research_collaboration": ["research_collaboration_formal", "research_collaboration_friendly"]
        }

        for family_name, templates in template_families.items():
            if template_key in templates:
                logger.debug("Template '%s' resolved to family '%s' for A/B alternatives", template_key, family_name)
                return [t for t in templates if t != template_key and t in self.templates]

        return []

    def _add_data_based_functions(self, extracted_data: dict[str, Any], selected_functions: list[str]) -> None:
        """Add personalization functions based on available data."""
        if extracted_data.get("dna_information"):
            dna_functions = ["dna_segment_analysis", "dna_ethnicity_correlation", "estimated_relationship", "shared_dna_amount"]
            best_dna_func = self._get_best_performing_function(dna_functions)
            if best_dna_func:
                selected_functions.append(best_dna_func)

        if extracted_data.get("locations"):
            location_functions = ["migration_pattern_context", "record_availability_assessment", "geographic_context"]
            best_location_func = self._get_best_performing_function(location_functions)
            if best_location_func:
                selected_functions.append(best_location_func)

        if extracted_data.get("vital_records"):
            historical_functions = ["historical_context_analysis", "generational_gap_analysis"]
            best_historical_func = self._get_best_performing_function(historical_functions)
            if best_historical_func:
                selected_functions.append(best_historical_func)

        if extracted_data.get("occupations"):
            selected_functions.append("occupation_social_context")

        if extracted_data.get("relationships"):
            selected_functions.append("family_size_analysis")

    def _add_advanced_functions(self, selected_functions: list[str]) -> None:
        """Add advanced functions based on effectiveness."""
        advanced_functions = ["surname_distribution_analysis", "document_preservation_likelihood"]
        for func in advanced_functions:
            if self._is_function_effective(func):
                selected_functions.append(func)

    def _select_optimal_personalization_functions(self, extracted_data: dict[str, Any]) -> list[str]:
        """Select optimal personalization functions based on data availability and effectiveness."""
        selected_functions = ["shared_ancestors", "genealogical_context", "research_focus"]

        self._add_data_based_functions(extracted_data, selected_functions)
        self._add_advanced_functions(selected_functions)

        return list(set(selected_functions))

    def _get_best_performing_function(self, function_list: list[str]) -> Optional[str]:
        """Get the best performing function from a list based on effectiveness data."""
        best_func = None
        best_score = 0.0

        for func in function_list:
            if func in self.effectiveness_tracker.effectiveness_data["personalization_effectiveness"]:
                stats = self.effectiveness_tracker.effectiveness_data["personalization_effectiveness"][func]
                if stats["usage_count"] >= 3:  # Minimum usage for consideration
                    score = stats["avg_engagement_score"]
                    if score > best_score:
                        best_score = score
                        best_func = func

        # If no function has enough data, return the first one
        return best_func or (function_list[0] if function_list else None)

    def _is_function_effective(self, function_name: str) -> bool:
        """Check if a personalization function is effective enough to use."""
        if function_name not in self.effectiveness_tracker.effectiveness_data["personalization_effectiveness"]:
            return True  # New functions get a chance

        stats = self.effectiveness_tracker.effectiveness_data["personalization_effectiveness"][function_name]
        if stats["usage_count"] < 5:
            return True  # Not enough data yet

        # Consider effective if positive response rate > 50% or avg engagement > 3.0
        positive_rate = stats["positive_responses"] / stats["usage_count"]
        return positive_rate > 0.5 or stats["avg_engagement_score"] > 3.0

    def _create_enhanced_format_data(
        self,
        extracted_data: dict[str, Any],
        base_format_data: dict[str, str],
        person_data: Optional[dict[str, Any]] = None,
        selected_functions: Optional[list[str]] = None
    ) -> dict[str, str]:
        """Create enhanced format data by applying selected personalization functions."""
        enhanced_data = base_format_data.copy()
        person_payload = person_data or {}

        self._apply_person_context(enhanced_data, person_payload)
        resolved_functions = self._resolve_selected_functions(selected_functions)
        self._apply_personalization_functions(resolved_functions, enhanced_data, extracted_data)
        self._ensure_required_template_keys(enhanced_data)

        return enhanced_data

    @staticmethod
    def _apply_person_context(enhanced_data: dict[str, str], person_payload: dict[str, Any]) -> None:
        """Populate person-specific defaults when available."""
        preferred_name = (
            person_payload.get("preferred_name")
            or person_payload.get("display_name")
            or person_payload.get("name")
            or person_payload.get("username")
        )
        if preferred_name:
            enhanced_data.setdefault("person_name", str(preferred_name))

        hometown = person_payload.get("home_location") or person_payload.get("location")
        if hometown:
            enhanced_data.setdefault("person_location", str(hometown))

    def _resolve_selected_functions(self, selected_functions: Optional[list[str]]) -> list[str]:
        """Determine which personalization functions should run."""
        if selected_functions is not None:
            return list(selected_functions)
        return list(self.personalization_functions_registry.keys())

    def _apply_personalization_functions(
        self,
        selected_functions: list[str],
        enhanced_data: dict[str, str],
        extracted_data: dict[str, Any],
    ) -> None:
        """Execute personalization functions safely."""
        for func_name in selected_functions:
            func = self.personalization_functions_registry.get(func_name)
            if func is None:
                continue
            try:
                enhanced_data[func_name] = func(extracted_data)
            except Exception as e:
                logger.warning(f"Error applying personalization function '{func_name}': {e}")
                enhanced_data[func_name] = self._get_fallback_value(func_name)

    def _ensure_required_template_keys(self, enhanced_data: dict[str, str]) -> None:
        """Guarantee essential template placeholders are populated."""
        for key, default_value in self._get_required_template_keys().items():
            enhanced_data.setdefault(key, default_value)

    @staticmethod
    def _get_required_template_keys() -> dict[str, str]:
        """Return required template values for backwards compatibility."""
        return {
            "total_rows": "many",
            "predicted_relationship": "family connection",
            "actual_relationship": "family connection",
            "relationship_path": "our shared family line",
        }

    @staticmethod
    def _get_fallback_value(func_name: str) -> str:
        """Get fallback value for a personalization function."""
        fallback_values = {
            "shared_ancestors": "our shared family line",
            "genealogical_context": "I'm excited to learn more about our family connection.",
            "research_focus": "our shared family history",
            "dna_segment_analysis": "DNA analysis could help us identify our connection.",
            "migration_pattern_context": "Understanding family migration helps trace our ancestry.",
            "historical_context_analysis": "Historical context provides valuable research insights.",
            "occupation_social_context": "Family occupations provide insights into our ancestors' lives."
        }
        return fallback_values.get(func_name, "our family research")

    def _format_shared_ancestors(self, extracted_data: dict[str, Any]) -> str:
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
        if len(ancestor_names) == 1:
            return ancestor_names[0]
        if len(ancestor_names) == 2:
            return f"{ancestor_names[0]} and {ancestor_names[1]}"
        return f"{', '.join(ancestor_names[:-1])}, and {ancestor_names[-1]}"

    @staticmethod
    def _format_single_vital_record(record: Any) -> Optional[str]:
        """Format a single vital record detail."""
        if not isinstance(record, dict):
            return None

        person = record.get("person", "")
        event_type = record.get("event_type", "")
        date = record.get("date", "")
        place = record.get("place", "")

        if not person or not (date or place):
            return None

        detail_parts = [person]
        if event_type and date:
            detail_parts.append(f"{event_type} {date}")
        if place:
            detail_parts.append(f"in {place}")

        return " ".join(detail_parts)

    def _format_ancestor_details(self, extracted_data: dict[str, Any]) -> str:
        """Format detailed ancestor information."""
        vital_records = extracted_data.get("vital_records", [])
        if not vital_records:
            return ""

        details = []
        for record in vital_records[:2]:
            formatted = self._format_single_vital_record(record)
            if formatted:
                details.append(formatted)

        return f" ({'; '.join(details)})" if details else ""

    def _create_genealogical_context(self, extracted_data: dict[str, Any]) -> str:
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

    @staticmethod
    def _format_occupations(occupations: list[Any]) -> str:
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

    @staticmethod
    def _identify_research_focus(extracted_data: dict[str, Any]) -> str:
        """Identify the main research focus from extracted data."""
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            # Use the first research question as focus
            return f" {research_questions[0].lower()}"

        # Fallback to general family history
        return " our shared family history"

    def _generate_specific_questions(self, extracted_data: dict[str, Any]) -> str:
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

    @staticmethod
    def _create_geographic_context(extracted_data: dict[str, Any]) -> str:
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

    def _format_location_context(self, extracted_data: dict[str, Any]) -> str:
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
            return f" {', '.join(location_names[:-1])}, and {location_names[-1]}"
        return ""

    @staticmethod
    def _create_research_suggestions(extracted_data: dict[str, Any]) -> str:
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

    @staticmethod
    def _format_research_questions(extracted_data: dict[str, Any]) -> str:
        """Format specific research questions."""
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            return f" - particularly about {research_questions[0].lower()}"
        return ""

    @staticmethod
    def _get_fallback_template(original_key: str) -> str:
        """Get a fallback template key based on the original key."""
        if "In_Tree" in original_key:
            return "In_Tree-Initial"
        if "Out_Tree" in original_key:
            return "Out_Tree-Initial"
        if "Productive" in original_key:
            return "Productive_Reply_Acknowledgement"
        return "In_Tree-Initial"  # Default fallback

    @staticmethod
    def _create_fallback_message(person_data: Optional[dict[str, Any]], base_format_data: dict[str, str]) -> str:
        """Create a simple fallback message when template processing fails."""
        name = base_format_data.get("name") or (person_data or {}).get("name") or "there"
        hometown = (person_data or {}).get("location") or "Aberdeen, Scotland"
        return (
            f"Dear {name},\n\nThank you for connecting! I'm excited to learn more about our family history."
            f"\n\nWarmest regards,\n\nWayne\n{hometown}"
        )

    # Additional helper methods for remaining format data fields
    @staticmethod
    def _format_mentioned_people(extracted_data: dict[str, Any]) -> str:
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
            if len(names) == 2:
                return f"{names[0]} and {names[1]}"
            return f"{', '.join(names[:-1])}, and {names[-1]}"
        return "your family history"

    @staticmethod
    def _create_research_context(extracted_data: dict[str, Any]) -> str:
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

    @staticmethod
    def _create_personalized_response(extracted_data: dict[str, Any]) -> str:
        """Create personalized response content."""
        structured_names = extracted_data.get("structured_names") or []
        target_name = "your family"
        if structured_names:
            first_entry = structured_names[0]
            if isinstance(first_entry, dict):
                target_name = first_entry.get("full_name") or first_entry.get("preferred_name") or target_name
            else:
                target_name = str(first_entry)

        research_focus = "our family connection"
        research_questions = extracted_data.get("research_questions") or []
        if research_questions:
            research_focus = research_questions[0]

        return (
            f"Your notes about {target_name} shed new light on {research_focus}. "
            "I appreciate the context you shared—it gives me several concrete next steps to pursue."
        )

    @staticmethod
    def _create_research_insights(extracted_data: dict[str, Any]) -> str:
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

    @staticmethod
    def _create_follow_up_questions(extracted_data: dict[str, Any]) -> str:
        """Create follow-up questions for continued research."""
        questions = []

        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            questions.append(f"Do you have any additional information about {research_questions[0].lower()}?")

        if not questions:
            questions.append("Do you have any other family documents or stories that might help our research?")

        return " ".join(questions)

    @staticmethod
    def _format_estimated_relationship(extracted_data: dict[str, Any]) -> str:
        """Format estimated relationship from DNA data."""
        dna_info = extracted_data.get("dna_information", [])
        for info in dna_info:
            if "cousin" in str(info).lower() or "relationship" in str(info).lower():
                return str(info)
        return "close family connection"

    @staticmethod
    def _format_shared_dna(extracted_data: dict[str, Any]) -> str:
        """Format shared DNA amount."""
        dna_info = extracted_data.get("dna_information", [])
        for info in dna_info:
            if "cm" in str(info).lower() or "centimorgans" in str(info).lower():
                return str(info)
        return "significant DNA"

    @staticmethod
    def _create_dna_context(extracted_data: dict[str, Any]) -> str:
        """Create DNA-specific context."""
        dna_info = extracted_data.get("dna_information", [])
        if dna_info:
            detail = str(dna_info[0])
            return (
                f"The DNA detail you mentioned ({detail}) helps narrow down our shared ancestors to just a few generations."
            )
        shared_cm = extracted_data.get("shared_dna_cm")
        if shared_cm:
            return f"Sharing roughly {shared_cm} cM usually indicates a fairly recent common ancestor."
        return "This DNA match suggests we have recent common ancestors worth exploring together."

    @staticmethod
    def _format_shared_ancestor_info(extracted_data: dict[str, Any]) -> str:
        """Format shared ancestor information."""
        shared_ancestors = extracted_data.get("shared_ancestors") or extracted_data.get("structured_names") or []
        names: list[str] = []
        for ancestor in shared_ancestors[:2]:
            if isinstance(ancestor, dict):
                full_name = ancestor.get("full_name")
                if full_name:
                    names.append(full_name)
            else:
                names.append(str(ancestor))
        if names:
            joined = " and ".join(names)
            return f"I'd love to compare our trees to confirm how we connect through {joined}."
        return "I'd love to compare our family trees to identify our common ancestors."

    @staticmethod
    def _create_collaboration_request(extracted_data: dict[str, Any]) -> str:
        """Create collaboration request text."""
        research_questions = extracted_data.get("research_questions") or []
        if research_questions:
            topic = research_questions[0].rstrip(".? ")
            return f"Would you be open to collaborating on solving the question about {topic}?"
        return "Would you be interested in collaborating on our genealogical research?"

    @staticmethod
    def _identify_research_topic(extracted_data: dict[str, Any]) -> str:
        """Identify the main research topic."""
        research_questions = extracted_data.get("research_questions", [])
        if research_questions:
            return research_questions[0]
        return "Family History Research"

    @staticmethod
    def _format_research_needs(extracted_data: dict[str, Any]) -> str:
        """Format specific research needs."""
        research_needs = extracted_data.get("research_needs") or extracted_data.get("research_questions") or []
        if research_needs:
            need = str(research_needs[0]).rstrip(". ")
            return f"I'm looking for additional documents or stories that might clarify {need}."
        return "I'm looking for additional information to complete our family history."

    @staticmethod
    def _create_collaboration_proposal(extracted_data: dict[str, Any]) -> str:
        """Create collaboration proposal text."""
        locations = extracted_data.get("locations", [])
        if locations and isinstance(locations[0], dict):
            place = locations[0].get("place")
            if place:
                return (
                    f"Perhaps we could share our findings about families in {place} and work together on any remaining mysteries."
                )
        return "Perhaps we could share our research findings and work together to solve any family history mysteries."

    # ========== NEW ADVANCED PERSONALIZATION FUNCTIONS ==========

    @staticmethod
    def _create_dna_segment_analysis(extracted_data: dict[str, Any]) -> str:
        """Create DNA segment analysis context for advanced users."""
        dna_info = extracted_data.get("dna_information", [])
        for info in dna_info:
            info_str = str(info).lower()
            if "segment" in info_str or "chromosome" in info_str:
                return f"The DNA segment data shows {info}, which helps narrow down our common ancestor timeframe."
        return "DNA segment analysis could help us identify our most recent common ancestor."

    @staticmethod
    def _create_migration_pattern_context(extracted_data: dict[str, Any]) -> str:
        """Create context about family migration patterns."""
        locations = extracted_data.get("locations", [])
        if len(locations) >= 2:
            places = []
            for loc in locations[:3]:
                if isinstance(loc, dict):
                    place = loc.get("place", "")
                    time_period = loc.get("time_period", "")
                    if place:
                        if time_period:
                            places.append(f"{place} ({time_period})")
                        else:
                            places.append(place)

            if len(places) >= 2:
                return f"I'm tracking family migration patterns from {places[0]} to {places[1]}, which aligns with historical movement patterns."

        return "Understanding family migration patterns helps piece together our shared ancestry."

    @staticmethod
    def _extract_year_from_date(date: str) -> Optional[int]:
        """Extract 4-digit year from date string."""
        for part in date.split():
            if part.isdigit() and len(part) == 4:
                year = int(part)
                if 1800 <= year <= 1950:
                    return year
        return None

    @staticmethod
    def _get_historical_context_for_location(year: int, place: str, event_type: str) -> Optional[str]:
        """Get historical context based on year and location."""
        if "Scotland" in place and 1840 <= year <= 1920:
            return f"The {event_type} in {place} around {year} coincides with significant Scottish emigration periods."
        if "Ireland" in place and 1845 <= year <= 1855:
            return f"The {event_type} in {place} around {year} was during the Irish Potato Famine era."
        if 1914 <= year <= 1918:
            return f"The {year} timeframe was during World War I, which affected many family records."
        return None

    def _analyze_vital_record_context(self, record: Any) -> Optional[str]:
        """Analyze historical context for a single vital record."""
        if not isinstance(record, dict):
            return None

        date = record.get("date", "")
        place = record.get("place", "")
        event_type = record.get("event_type", "")

        if not (date and place):
            return None

        year = self._extract_year_from_date(date)
        if year:
            return self._get_historical_context_for_location(year, place, event_type)
        return None

    def _create_historical_context_analysis(self, extracted_data: dict[str, Any]) -> str:
        """Create historical context based on dates and locations."""
        vital_records = extracted_data.get("vital_records", [])

        for record in vital_records[:2]:
            context = self._analyze_vital_record_context(record)
            if context:
                return context

        return "Understanding the historical context of our family events helps explain migration and life decisions."

    @staticmethod
    def _create_record_availability_assessment(extracted_data: dict[str, Any]) -> str:
        """Assess likely record availability based on location and time period."""
        locations = extracted_data.get("locations", [])

        for location in locations:
            if isinstance(location, dict):
                place = location.get("place", "")

                if "Scotland" in place:
                    return "Scottish records are generally well-preserved, especially civil registration after 1855 and parish records."
                if "Ireland" in place:
                    return "Irish records can be challenging due to the 1922 Public Record Office fire, but many alternatives exist."
                if "England" in place:
                    return "English records are extensive, with civil registration from 1837 and excellent parish records."
                if "Poland" in place or "Ukraine" in place:
                    return "Eastern European records require specialized research due to border changes and wartime losses."

        return "Record availability varies by location and time period - I can help identify the best sources for our research."

    @staticmethod
    def _create_dna_ethnicity_correlation(extracted_data: dict[str, Any]) -> str:
        """Create correlation between DNA ethnicity and documented ancestry."""
        dna_info = extracted_data.get("dna_information", [])
        locations = extracted_data.get("locations", [])

        ethnicity_regions = []
        documented_regions = []

        for info in dna_info:
            info_str = str(info).lower()
            if any(region in info_str for region in ["scottish", "irish", "english", "polish", "ukrainian"]):
                ethnicity_regions.append(info_str)

        for location in locations:
            if isinstance(location, dict):
                place = location.get("place", "")
                if place:
                    documented_regions.append(place.lower())

        if ethnicity_regions and documented_regions:
            return "The DNA ethnicity results align well with our documented family locations, confirming our research direction."

        return "Comparing DNA ethnicity with documented ancestry helps validate our family tree research."

    @staticmethod
    def _create_surname_distribution_analysis(extracted_data: dict[str, Any]) -> str:
        """Analyze surname distribution patterns."""
        names = extracted_data.get("structured_names", [])
        locations = extracted_data.get("locations", [])

        surnames = []
        places = []

        for name in names:
            if isinstance(name, dict):
                full_name = name.get("full_name", "")
                if full_name and " " in full_name:
                    surname = full_name.split()[-1]
                    surnames.append(surname)

        for location in locations:
            if isinstance(location, dict):
                place = location.get("place", "")
                if place:
                    places.append(place)

        if surnames and places:
            surname = surnames[0]
            place = places[0]
            return f"The {surname} surname distribution in {place} can help identify other family branches and potential connections."

        return "Surname distribution patterns often reveal family migration routes and concentrations."

    @staticmethod
    def _get_occupation_context(occupation: str, person: str) -> Optional[str]:
        """Get social context for a specific occupation."""
        occupation_lower = occupation.lower()

        occupation_contexts = {
            ("farmer", "agricultural"): f"{person}'s agricultural work suggests rural family roots, which often means strong community ties and local records.",
            ("miner", "mining"): f"{person}'s mining work indicates industrial family history, often with company records and mining community connections.",
            ("fisherman", "fishing"): f"{person}'s fishing occupation suggests coastal family traditions and maritime community connections.",
            ("teacher", "educator"): f"{person}'s teaching profession indicates educated family background with potential school and community records.",
            ("merchant", "trader"): f"{person}'s merchant work suggests business connections and potential commercial records.",
        }

        for keywords, context in occupation_contexts.items():
            if any(keyword in occupation_lower for keyword in keywords):
                return context
        return None

    def _create_occupation_social_context(self, extracted_data: dict[str, Any]) -> str:
        """Create social context based on occupations."""
        occupations = extracted_data.get("occupations", [])

        for occ in occupations:
            if isinstance(occ, dict):
                occupation = occ.get("occupation", "")
                person = occ.get("person", "")
                context = self._get_occupation_context(occupation, person)
                if context:
                    return context

        return "Family occupations provide insights into social status, community connections, and available records."

    @staticmethod
    def _create_family_size_analysis(extracted_data: dict[str, Any]) -> str:
        """Analyze family size patterns and implications."""
        relationships = extracted_data.get("relationships", [])

        children_count = 0
        siblings_count = 0

        for rel in relationships:
            if isinstance(rel, dict):
                relationship = rel.get("relationship", "").lower()
                if relationship == "child":
                    children_count += 1
                elif relationship == "sibling":
                    siblings_count += 1

        if children_count >= 6:
            return f"Large families of {children_count}+ children were common in that era and often indicate strong family traditions and extensive cousin networks."
        if children_count >= 3:
            return f"The family size of {children_count} children suggests good survival rates and potential for extensive descendant research."
        if siblings_count >= 4:
            return f"Large sibling groups of {siblings_count}+ often mean multiple family lines to research and potential DNA matches through various branches."

        return "Family size patterns help predict the scope of potential DNA matches and research opportunities."

    # Note: _extract_year_from_date is defined earlier in the class (line 766)

    def _extract_birth_and_marriage_years(self, vital_records: list[Any]) -> tuple[list[int], list[int]]:
        """Extract birth and marriage years from vital records."""
        birth_years = []
        marriage_years = []

        for record in vital_records:
            if isinstance(record, dict):
                event_type = record.get("event_type", "")
                date = record.get("date", "")
                year = self._extract_year_from_date(date)

                if year:
                    if event_type == "birth":
                        birth_years.append(year)
                    elif event_type == "marriage":
                        marriage_years.append(year)

        return birth_years, marriage_years

    @staticmethod
    def _analyze_birth_year_gap(birth_years: list[int]) -> Optional[str]:
        """Analyze gap between birth years and return interpretation."""
        if len(birth_years) >= 2:
            gap = abs(birth_years[0] - birth_years[1])
            if gap >= 25:
                return f"The {gap}-year generational gap suggests we may be looking at parent-child relationships, which helps narrow our connection."
            if gap <= 10:
                return f"The {gap}-year age gap suggests sibling or cousin relationships, indicating we share more recent common ancestors."
        return None

    def _create_generational_gap_analysis(self, extracted_data: dict[str, Any]) -> str:
        """Analyze generational gaps and their implications."""
        vital_records = extracted_data.get("vital_records", [])
        birth_years, _ = self._extract_birth_and_marriage_years(vital_records)

        analysis = self._analyze_birth_year_gap(birth_years)
        if analysis:
            return analysis

        return "Analyzing generational gaps helps estimate relationship distances and common ancestor timeframes."

    @staticmethod
    def _create_document_preservation_likelihood(extracted_data: dict[str, Any]) -> str:
        """Assess likelihood of document preservation based on context."""
        locations = extracted_data.get("locations", [])

        for location in locations:
            if isinstance(location, dict):
                place = location.get("place", "")
                context = location.get("context", "")

                if "church" in context.lower() or "parish" in context.lower():
                    return f"Church connections in {place} suggest good preservation of parish records and potential for baptism, marriage, and burial documentation."
                if "military" in context.lower():
                    return f"Military service records from {place} are often well-preserved and can provide detailed family information."
                if "immigration" in context.lower() or "emigration" in context.lower():
                    return f"Immigration records for {place} often include family details and can help trace origins and destinations."

        return "Document preservation varies by location and institution - I can help identify the most promising record sources."


class MessageEffectivenessTracker:
    """
    Advanced message effectiveness tracking and analytics system.
    Tracks response rates, engagement quality, and optimization opportunities.
    """

    def __init__(self) -> None:
        """Initialize the effectiveness tracker."""
        self.effectiveness_data = self._load_effectiveness_data()
        self.response_categories = {
            "ENTHUSIASTIC": 5,
            "PRODUCTIVE": 4,
            "CAUTIOUSLY_INTERESTED": 3,
            "CONFUSED": 2,
            "UNINTERESTED": 1,
            "DESIST": 0,
            "OTHER": 2
        }

    @staticmethod
    def _load_effectiveness_data() -> dict[str, Any]:
        """Load existing effectiveness tracking data."""
        try:
            script_dir = Path(__file__).resolve().parent
            effectiveness_path = script_dir / "message_effectiveness.json"

            if effectiveness_path.exists():
                with effectiveness_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {
                    "template_performance": {},
                    "personalization_effectiveness": {},
                    "response_analytics": {},
                    "optimization_insights": {},
                    "last_updated": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error loading effectiveness data: {e}")
            return {"template_performance": {}, "personalization_effectiveness": {}}

    def track_message_response(
        self,
        template_key: str,
        personalization_functions_used: list[str],
        response_intent: str,
        response_quality_score: float,
        conversation_length: int,
        genealogical_data_extracted: int
    ) -> None:
        """
        Track the effectiveness of a message and its response.

        Args:
            template_key: The message template used
            personalization_functions_used: List of personalization functions applied
            response_intent: The classified intent of the response
            response_quality_score: Quality score of the response (0-10)
            conversation_length: Number of messages in conversation
            genealogical_data_extracted: Amount of genealogical data extracted
        """
        try:
            # Update template performance
            if template_key not in self.effectiveness_data["template_performance"]:
                self.effectiveness_data["template_performance"][template_key] = {
                    "total_sent": 0,
                    "responses_received": 0,
                    "avg_response_quality": 0.0,
                    "intent_distribution": {},
                    "avg_data_extracted": 0.0,
                    "avg_conversation_length": 0.0,
                }

            template_stats = self.effectiveness_data["template_performance"][template_key]
            template_stats.setdefault("avg_conversation_length", 0.0)
            template_stats["total_sent"] += 1
            total_sent = template_stats["total_sent"]
            current_avg_length = template_stats["avg_conversation_length"]
            normalized_length = max(conversation_length, 0)
            template_stats["avg_conversation_length"] = (
                (current_avg_length * (total_sent - 1) + normalized_length) / total_sent
            )

            if response_intent != "NO_RESPONSE":
                template_stats["responses_received"] += 1

                # Update intent distribution
                if response_intent not in template_stats["intent_distribution"]:
                    template_stats["intent_distribution"][response_intent] = 0
                template_stats["intent_distribution"][response_intent] += 1

                # Update average response quality
                current_avg = template_stats["avg_response_quality"]
                response_count = template_stats["responses_received"]
                template_stats["avg_response_quality"] = (
                    (current_avg * (response_count - 1) + response_quality_score) / response_count
                )

                # Update average data extracted
                current_data_avg = template_stats["avg_data_extracted"]
                template_stats["avg_data_extracted"] = (
                    (current_data_avg * (response_count - 1) + genealogical_data_extracted) / response_count
                )

            # Track personalization function effectiveness
            for func_name in personalization_functions_used:
                if func_name not in self.effectiveness_data["personalization_effectiveness"]:
                    self.effectiveness_data["personalization_effectiveness"][func_name] = {
                        "usage_count": 0,
                        "positive_responses": 0,
                        "avg_engagement_score": 0.0
                    }

                func_stats = self.effectiveness_data["personalization_effectiveness"][func_name]
                func_stats["usage_count"] += 1

                # Consider ENTHUSIASTIC, PRODUCTIVE, CAUTIOUSLY_INTERESTED as positive
                if response_intent in {"ENTHUSIASTIC", "PRODUCTIVE", "CAUTIOUSLY_INTERESTED"}:
                    func_stats["positive_responses"] += 1

                # Update engagement score
                engagement_score = self.response_categories.get(response_intent, 2)
                current_avg = func_stats["avg_engagement_score"]
                usage_count = func_stats["usage_count"]
                func_stats["avg_engagement_score"] = (
                    (current_avg * (usage_count - 1) + engagement_score) / usage_count
                )

            # Update last modified
            self.effectiveness_data["last_updated"] = datetime.now().isoformat()

            # Save updated data
            self._save_effectiveness_data()

            logger.info(f"Tracked message effectiveness for template '{template_key}' with response '{response_intent}'")

        except Exception as e:
            logger.error(f"Error tracking message response: {e}")

    def get_template_effectiveness_score(self, template_key: str) -> float:
        """Get effectiveness score for a specific template (0-10)."""
        if template_key not in self.effectiveness_data["template_performance"]:
            return 5.0  # Default neutral score

        stats = self.effectiveness_data["template_performance"][template_key]

        if stats["total_sent"] == 0:
            return 5.0

        # Calculate response rate (0-4 points)
        response_rate = stats["responses_received"] / stats["total_sent"]
        response_score = response_rate * 4

        # Add quality bonus (0-3 points)
        quality_score = (stats["avg_response_quality"] / 10) * 3

        # Add data extraction bonus (0-3 points)
        data_score = min(stats["avg_data_extracted"] / 5, 1) * 3

        return min(response_score + quality_score + data_score, 10.0)

    def get_optimization_recommendations(self) -> list[str]:
        """Get recommendations for optimizing message effectiveness."""
        recommendations = []

        # Analyze template performance
        template_scores = {}
        for template_key in self.effectiveness_data["template_performance"]:
            template_scores[template_key] = self.get_template_effectiveness_score(template_key)

        if template_scores:
            best_template = max(template_scores.keys(), key=lambda k: template_scores[k])
            worst_template = min(template_scores.keys(), key=lambda k: template_scores[k])

            if template_scores[best_template] > 7.0:
                recommendations.append(f"Template '{best_template}' is performing excellently (score: {template_scores[best_template]:.1f}). Consider using it more frequently.")

            if template_scores[worst_template] < 4.0:
                recommendations.append(f"Template '{worst_template}' needs improvement (score: {template_scores[worst_template]:.1f}). Consider revising or replacing.")

        # Analyze personalization function effectiveness
        func_effectiveness = {}
        for func_name, stats in self.effectiveness_data["personalization_effectiveness"].items():
            if stats["usage_count"] >= 5:  # Only consider functions used at least 5 times
                effectiveness = stats["positive_responses"] / stats["usage_count"]
                func_effectiveness[func_name] = effectiveness

        if func_effectiveness:
            best_func = max(func_effectiveness.keys(), key=lambda k: func_effectiveness[k])
            worst_func = min(func_effectiveness.keys(), key=lambda k: func_effectiveness[k])

            if func_effectiveness[best_func] > 0.7:
                recommendations.append(f"Personalization function '{best_func}' is highly effective ({func_effectiveness[best_func]:.1%} positive response rate). Use it more often.")

            if func_effectiveness[worst_func] < 0.3:
                recommendations.append(f"Personalization function '{worst_func}' has low effectiveness ({func_effectiveness[worst_func]:.1%} positive response rate). Consider improving or replacing.")

        return recommendations

    def _save_effectiveness_data(self) -> None:
        """Save effectiveness data to file."""
        try:
            script_dir = Path(__file__).resolve().parent
            effectiveness_path = script_dir / "message_effectiveness.json"

            with effectiveness_path.open("w", encoding="utf-8") as f:
                json.dump(self.effectiveness_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving effectiveness data: {e}")


# Test functions
def test_message_personalization() -> bool:
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
    message, _ = personalizer.create_personalized_message(
        "Enhanced_In_Tree-Initial",
        test_person_data,
        test_extracted_data,
        test_base_data
    )

    success = len(message) > 100 and "John Smith" in message
    logger.info(f"Message personalization test: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def test_fallback_template_path() -> bool:
    """Ensure an unknown template key triggers safe fallback without exception."""
    personalizer = MessagePersonalizer()
    # Force empty templates to guarantee fallback path
    personalizer.templates = {"In_Tree-Initial": "Hello {name}!"}
    msg, _ = personalizer.create_personalized_message(
        "Totally_Unknown_Template",
        {"username": "UserX"},
        {},
        {"name": "UserX"}
    )
    # Either fallback message or resolved fallback template must appear
    return ("UserX" in msg) and len(msg) > 10


def test_shared_ancestors_formatting() -> bool:
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


def test_location_context_limit() -> bool:
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


def _test_personalizer_initialization():
    """Test MessagePersonalizer initialization."""
    personalizer = MessagePersonalizer()

    assert personalizer is not None, "Personalizer should be created"
    assert hasattr(personalizer, 'templates'), "Should have templates"
    assert hasattr(personalizer, 'personalization_config'), "Should have config"
    assert hasattr(personalizer, 'effectiveness_tracker'), "Should have tracker"

    logger.info("✓ MessagePersonalizer initialized correctly")
    return True


def _test_personalization_config():
    """Test personalization configuration loading."""
    personalizer = MessagePersonalizer()
    config = personalizer.personalization_config

    assert isinstance(config, dict), "Config should be dictionary"
    assert 'max_ancestors_to_mention' in config, "Should have max_ancestors setting"
    assert 'max_locations_to_mention' in config, "Should have max_locations setting"
    assert 'max_research_questions' in config, "Should have max_questions setting"

    logger.info("✓ Personalization config loaded correctly")
    return True


def _test_personalization_registry():
    """Test personalization functions registry."""
    personalizer = MessagePersonalizer()
    registry = personalizer.personalization_functions_registry

    assert isinstance(registry, dict), "Registry should be dictionary"
    assert len(registry) > 0, "Registry should have functions"
    assert 'shared_ancestors' in registry, "Should have shared_ancestors function"
    assert 'geographic_context' in registry, "Should have geographic_context function"

    logger.info("✓ Personalization registry built correctly")
    return True


def _test_effectiveness_tracker_initialization():
    """Test MessageEffectivenessTracker initialization."""
    tracker = MessageEffectivenessTracker()

    assert tracker is not None, "Tracker should be created"
    assert hasattr(tracker, 'effectiveness_data'), "Should have effectiveness_data"
    assert hasattr(tracker, 'response_categories'), "Should have response_categories"

    logger.info("✓ MessageEffectivenessTracker initialized correctly")
    return True


def _test_shared_ancestors_formatting_empty():
    """Test shared ancestors formatting with empty data."""
    personalizer = MessagePersonalizer()
    formatted = personalizer._format_shared_ancestors({})

    assert isinstance(formatted, str), "Should return string"
    assert len(formatted) > 0, "Should return non-empty string"

    logger.info("✓ Empty shared ancestors formatted correctly")
    return True


def _test_location_context_formatting_empty():
    """Test location context formatting with empty data."""
    personalizer = MessagePersonalizer()
    formatted = personalizer._format_location_context({})

    assert isinstance(formatted, str), "Should return string"
    # Empty data returns empty string - this is expected behavior
    assert not formatted, "Should return empty string for empty data"

    logger.info("✓ Empty location context formatted correctly")
    return True


def _test_dna_context_creation():
    """Test DNA context creation."""
    personalizer = MessagePersonalizer()
    context = personalizer._create_dna_context({'shared_dna_cm': 212.0, 'relationship': '2nd cousin'})

    assert isinstance(context, str), "Should return string"

    logger.info("✓ DNA context created correctly")
    return True


def _test_null_and_none_inputs() -> None:
    """Test personalizer handles None and null inputs gracefully (negative path)."""
    personalizer = MessagePersonalizer()

    # Test with all None inputs - should handle gracefully or raise expected error
    try:
        message, functions_used = personalizer.create_personalized_message(
            "Enhanced_In_Tree-Initial",
            None,
            None,
            None,
        )
        # If no exception, verify fallback behavior
        assert isinstance(message, str), "Should return string even with None inputs"
        assert len(message) > 0, "Should return non-empty fallback message for None inputs"
        assert isinstance(functions_used, list), "Should return list of functions even with None inputs"
    except (AttributeError, TypeError) as e:
        # If it raises an error, that's acceptable for None inputs
        # The important thing is we documented the behavior
        assert "NoneType" in str(e), f"Should raise descriptive error for None inputs, got: {e}"

    # Test with empty dicts (this should work)
    message, _ = personalizer.create_personalized_message(
        "Enhanced_In_Tree-Initial",
        {},
        {},
        {}
    )
    assert isinstance(message, str), "Should return string with empty dicts"
    assert len(message) > 0, "Should return non-empty message with empty dicts"


def _test_malformed_extracted_data() -> None:
    """Test with malformed structured data in extracted_data."""
    personalizer = MessagePersonalizer()

    # Test with invalid structured_names (not a list of dicts)
    malformed_data = {
        "structured_names": "not a list",  # Should be list of dicts
        "vital_records": [123, 456],  # Invalid record types
        "locations": ["string instead of dict"],  # Should be list of dicts
        "research_questions": None,  # Should be list
    }

    message, _ = personalizer.create_personalized_message(
        "Enhanced_In_Tree-Initial",
        {"username": "Test"},
        malformed_data,
        {"name": "Test"}
    )

    assert isinstance(message, str), "Should handle malformed data without crashing"
    assert len(message) > 0, "Should return non-empty message despite malformed data"
    assert "Test" in message, "Should still use base format data when extraction fails"


def _test_unicode_and_special_characters() -> None:
    """Test message personalization with Unicode and special characters."""
    personalizer = MessagePersonalizer()

    unicode_data = {
        "structured_names": [
            {"full_name": "José María García-López"},
            {"full_name": "Müller, Jürgen"},
            {"full_name": "Владимир Путин"},
            {"full_name": "李明"},
            {"full_name": "O'Connor-Władysław"}
        ],
        "vital_records": [
            {"person": "José María", "event_type": "birth", "date": "1850", "place": "Málaga, España"}
        ],
        "locations": [
            {"place": "Köln, Deutschland", "context": "residence", "time_period": "1900s"}
        ]
    }

    message, _ = personalizer.create_personalized_message(
        "Enhanced_In_Tree-Initial",
        {"username": "Tëst Üsér"},
        unicode_data,
        {"name": "Tëst Üsér"}
    )

    assert isinstance(message, str), "Should handle Unicode characters"
    assert len(message) > 0, "Should generate message with Unicode names"
    # Verify some Unicode preserved in message
    unicode_chars_present = any(char in message for char in "ÁéíóúüñÖäМирЛ明")
    assert unicode_chars_present or "José" in message, "Should preserve or safely handle Unicode in output"


def _test_extremely_long_inputs() -> None:
    """Test with extremely long names and text fields."""
    personalizer = MessagePersonalizer()

    # Create extremely long name (500 characters)
    long_name = "A" * 500
    very_long_place = "B" * 1000

    long_data = {
        "structured_names": [{"full_name": long_name}] * 50,  # 50 extremely long names
        "vital_records": [
            {"person": long_name, "event_type": "birth", "date": "1800", "place": very_long_place}
        ] * 20,  # 20 records with long data
        "locations": [{"place": very_long_place, "context": "residence"}] * 30
    }

    message, _ = personalizer.create_personalized_message(
        "Enhanced_In_Tree-Initial",
        {"username": "Test"},
        long_data,
        {"name": "Test"}
    )

    assert isinstance(message, str), "Should handle extremely long inputs"
    assert len(message) > 0, "Should generate message despite long inputs"
    # Message should not be absurdly long (personalization should limit)
    assert len(message) < 100000, "Message length should be reasonable despite long inputs"


def _test_missing_template_keys() -> None:
    """Test message generation with missing template placeholder keys."""
    personalizer = MessagePersonalizer()

    # Create template with many placeholders
    complex_template = """
    Dear {name},

    {shared_ancestors} {ancestor_details} {genealogical_context}
    {geographic_context} {research_focus} {specific_questions}
    {missing_key_1} {missing_key_2} {missing_key_3}

    Best, Wayne
    """

    personalizer.templates["test_complex"] = complex_template

    # Provide minimal format data (missing most keys)
    message, _ = personalizer.create_personalized_message(
        "test_complex",
        {"username": "Test"},
        {"structured_names": []},
        {"name": "Test User"}
    )

    assert isinstance(message, str), "Should handle missing template keys"
    assert len(message) > 0, "Should generate message despite missing keys"
    assert "Test User" in message, "Should use provided keys successfully"
    # Should not contain unreplaced placeholders
    assert "{missing_key" not in message, "Should replace or remove missing placeholder keys"


def _test_zero_and_negative_numbers() -> None:
    """Test with zero and negative numbers in DNA/relationship data."""
    personalizer = MessagePersonalizer()

    edge_case_data = {
        "shared_dna_cm": 0.0,  # Zero DNA
        "relationship": "unknown",
        "structured_names": [],
        "birth_year": -100,  # Invalid negative year
        "total_rows": -5,  # Negative count
    }

    message, _ = personalizer.create_personalized_message(
        "Enhanced_In_Tree-Initial",
        {"username": "Test"},
        edge_case_data,
        {"name": "Test", "total_rows": "0"}
    )

    assert isinstance(message, str), "Should handle zero and negative numbers"
    assert len(message) > 0, "Should generate message with edge case numbers"


def _test_format_single_vital_record_edge_cases() -> None:
    """Test _format_single_vital_record with various edge cases."""
    personalizer = MessagePersonalizer()

    # Test with None
    result = personalizer._format_single_vital_record(None)
    assert result is None, "Should return None for None input"

    # Test with non-dict
    result = personalizer._format_single_vital_record("not a dict")
    assert result is None, "Should return None for non-dict input"

    # Test with empty dict
    result = personalizer._format_single_vital_record({})
    assert result is None, "Should return None for empty dict"

    # Test with missing required fields
    result = personalizer._format_single_vital_record({"person": "John"})
    assert result is None, "Should return None when date and place both missing"

    # Test with only date
    result = personalizer._format_single_vital_record({
        "person": "John Smith",
        "event_type": "birth",
        "date": "1850"
    })
    assert isinstance(result, str), "Should format with date only"
    assert "John Smith" in result, "Should include person name"
    assert "1850" in result, "Should include date"


def message_personalization_module_tests() -> bool:
    """
    Comprehensive test suite for message_personalization.py with real functionality testing.
    Tests message personalization, template generation, and intelligent messaging systems.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Message Personalization & Template Generation", "message_personalization.py")
    suite.start_suite()

    with suppress_logging():
        # Initialization tests
        suite.run_test(
            "Personalizer initialization",
            _test_personalizer_initialization,
            "MessagePersonalizer initialized correctly",
            "Test MessagePersonalizer initialization",
            "Verify personalizer creates with all required attributes"
        )

        suite.run_test(
            "Personalization config",
            _test_personalization_config,
            "Personalization config loaded correctly",
            "Test personalization configuration loading",
            "Verify config has all required settings"
        )

        suite.run_test(
            "Personalization registry",
            _test_personalization_registry,
            "Personalization registry built correctly",
            "Test personalization functions registry",
            "Verify registry has all required functions"
        )

        suite.run_test(
            "Effectiveness tracker initialization",
            _test_effectiveness_tracker_initialization,
            "MessageEffectivenessTracker initialized correctly",
            "Test MessageEffectivenessTracker initialization",
            "Verify tracker creates with required attributes"
        )

        # Edge case tests
        suite.run_test(
            "Shared ancestors formatting empty",
            _test_shared_ancestors_formatting_empty,
            "Empty shared ancestors formatted correctly",
            "Test shared ancestors formatting with empty list",
            "Verify graceful handling of empty ancestor list"
        )

        suite.run_test(
            "Location context formatting empty",
            _test_location_context_formatting_empty,
            "Empty location context formatted correctly",
            "Test location context formatting with empty list",
            "Verify graceful handling of empty location list"
        )

        suite.run_test(
            "DNA context creation",
            _test_dna_context_creation,
            "DNA context created correctly",
            "Test DNA context creation",
            "Verify DNA context formatting with sample data"
        )

        # Negative path tests
        suite.run_test(
            "Null and None inputs",
            _test_null_and_none_inputs,
            "Handles None/null inputs gracefully",
            "Test with None person_data, extracted_data, base_format_data",
            "Verify no crashes on None inputs and fallback message generation"
        )

        suite.run_test(
            "Malformed extracted data",
            _test_malformed_extracted_data,
            "Handles malformed structured data",
            "Test with invalid structured_names, vital_records, locations",
            "Verify resilience to incorrect data types in extraction results"
        )

        suite.run_test(
            "Unicode and special characters",
            _test_unicode_and_special_characters,
            "Preserves/handles Unicode characters",
            "Test with names like José María, Müller, Владимир, 李明, O'Connor-Władysław",
            "Verify internationalization support and special character handling"
        )

        suite.run_test(
            "Extremely long inputs",
            _test_extremely_long_inputs,
            "Handles extremely long names and text",
            "Test with 500-char names, 1000-char places, 50+ records",
            "Verify reasonable message length limits despite large input data"
        )

        suite.run_test(
            "Missing template keys",
            _test_missing_template_keys,
            "Handles missing template placeholder keys",
            "Test with template containing {missing_key_1}, {missing_key_2}",
            "Verify no unreplaced placeholders in output when keys missing"
        )

        suite.run_test(
            "Zero and negative numbers",
            _test_zero_and_negative_numbers,
            "Handles zero/negative numbers in data",
            "Test with shared_dna_cm=0.0, birth_year=-100, total_rows=-5",
            "Verify edge case numeric values don't cause crashes"
        )

        suite.run_test(
            "Format single vital record edge cases",
            _test_format_single_vital_record_edge_cases,
            "_format_single_vital_record handles edge cases",
            "Test with None, non-dict, empty dict, missing fields",
            "Verify proper None returns and graceful degradation for invalid inputs"
        )

        # Original tests
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


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(message_personalization_module_tests)


if __name__ == "__main__":
    """
    Execute comprehensive message personalization tests when run directly.
    Tests message personalization, template generation, and intelligent messaging systems.
    """
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
