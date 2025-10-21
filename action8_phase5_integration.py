"""
Action 8 Phase 5 Integration

Integrates Phase 5 Research Assistant Features into Action 8 messaging:
- Source citation extraction from GEDCOM
- Research suggestion generation
- Relationship diagram generation
- Record sharing capabilities
- AI-powered research guidance prompts

This module enhances message content with genealogical research capabilities.
"""

from typing import Any, Optional

from database import DnaMatch, FamilyTree, Person
from gedcom_utils import format_source_citations, get_person_sources
from record_sharing import create_record_sharing_message, format_record_reference
from relationship_diagram import format_relationship_for_message
from research_guidance_prompts import create_research_guidance_prompt
from research_suggestions import generate_research_suggestions
from standard_imports import *

logger = logging.getLogger(__name__)


def enhance_message_with_sources(
    person: Person,
    family_tree: Optional[FamilyTree],
    format_data: dict[str, Any]
) -> None:
    """
    Enhance message format data with source citations from GEDCOM.
    
    Args:
        person: Person being messaged
        family_tree: Family tree relationship (if in tree)
        format_data: Message format data to enhance (modified in place)
    """
    if not family_tree or not hasattr(family_tree, 'gedcom_id'):
        format_data['source_citations'] = ""
        return

    try:
        from config_manager import config
        gedcom_file = config.database.gedcom_file_path

        if not gedcom_file or not os.path.exists(gedcom_file):
            format_data['source_citations'] = ""
            return

        # Load GEDCOM and get sources
        from gedcom_utils import load_gedcom_data
        gedcom_data = load_gedcom_data(gedcom_file)

        if not gedcom_data or not hasattr(gedcom_data, 'indi_index'):
            format_data['source_citations'] = ""
            return

        gedcom_id = family_tree.gedcom_id
        if gedcom_id not in gedcom_data.indi_index:
            format_data['source_citations'] = ""
            return

        individual = gedcom_data.indi_index[gedcom_id]
        sources = get_person_sources(individual)

        if sources and any(sources.values()):
            citation = format_source_citations(sources)
            format_data['source_citations'] = f" They are {citation}."
        else:
            format_data['source_citations'] = ""

    except Exception as e:
        logger.debug(f"Could not extract sources for {person.username}: {e}")
        format_data['source_citations'] = ""


def enhance_message_with_relationship_diagram(
    person: Person,
    family_tree: Optional[FamilyTree],
    format_data: dict[str, Any]
) -> None:
    """
    Enhance message format data with relationship diagram.
    
    Args:
        person: Person being messaged
        family_tree: Family tree relationship (if in tree)
        format_data: Message format data to enhance (modified in place)
    """
    if not family_tree or not hasattr(family_tree, 'relationship_path'):
        format_data['relationship_diagram'] = ""
        return

    try:
        relationship_path = family_tree.relationship_path
        if not relationship_path:
            format_data['relationship_diagram'] = ""
            return

        # Parse relationship path (stored as JSON string)
        import json
        if isinstance(relationship_path, str):
            path = json.loads(relationship_path)
        else:
            path = relationship_path

        if not path or len(path) < 2:
            format_data['relationship_diagram'] = ""
            return

        # Generate compact diagram for messages (not too long)
        from_name = "You"
        to_name = person.first_name or person.username or "them"

        diagram_text = format_relationship_for_message(
            from_name,
            to_name,
            path,
            include_diagram=True,
            style="compact"
        )

        format_data['relationship_diagram'] = f"\n\n{diagram_text}"

    except Exception as e:
        logger.debug(f"Could not generate relationship diagram for {person.username}: {e}")
        format_data['relationship_diagram'] = ""


def enhance_message_with_research_suggestions(
    person: Person,
    family_tree: Optional[FamilyTree],
    format_data: dict[str, Any]
) -> None:
    """
    Enhance message format data with research suggestions.
    
    Args:
        person: Person being messaged
        family_tree: Family tree relationship (if in tree)
        format_data: Message format data to enhance (modified in place)
    """
    try:
        # Extract location and time period information
        locations = []
        time_periods = []
        common_ancestors = []

        # Get birth/death info from person
        if hasattr(person, 'birth_year') and person.birth_year:
            decade = (person.birth_year // 10) * 10
            time_periods.append(f"{decade}s")

        # Get common ancestor info from family tree
        if family_tree and hasattr(family_tree, 'common_ancestor_name'):
            ancestor_name = family_tree.common_ancestor_name
            if ancestor_name:
                common_ancestors.append({
                    'name': ancestor_name,
                    'birth_year': None,
                    'birth_place': None
                })

        # Only generate suggestions if we have enough context
        if not (locations or time_periods or common_ancestors):
            format_data['research_suggestions'] = ""
            return

        # Generate suggestions
        result = generate_research_suggestions(
            common_ancestors=common_ancestors if common_ancestors else [{}],
            locations=locations if locations else [""],
            time_periods=time_periods if time_periods else [""]
        )

        collections = result.get('collections', [])
        if collections:
            # Format top 2 suggestions for message
            top_suggestions = collections[:2]
            suggestions_text = "\n\nResearch suggestions:\n"
            for i, coll in enumerate(top_suggestions, 1):
                suggestions_text += f"{i}. {coll.get('name', 'Unknown collection')}\n"
            format_data['research_suggestions'] = suggestions_text
        else:
            format_data['research_suggestions'] = ""

    except Exception as e:
        logger.debug(f"Could not generate research suggestions for {person.username}: {e}")
        format_data['research_suggestions'] = ""


def enhance_message_format_data_phase5(
    person: Person,
    family_tree: Optional[FamilyTree],
    dna_match: Optional[DnaMatch],
    format_data: dict[str, Any],
    enable_sources: bool = True,
    enable_diagrams: bool = True,
    enable_suggestions: bool = False
) -> None:
    """
    Enhance message format data with all Phase 5 features.
    
    This is the main integration point for Action 8. Call this function
    after preparing base format data to add Phase 5 enhancements.
    
    Args:
        person: Person being messaged
        family_tree: Family tree relationship (if in tree)
        dna_match: DNA match data
        format_data: Message format data to enhance (modified in place)
        enable_sources: Whether to add source citations
        enable_diagrams: Whether to add relationship diagrams
        enable_suggestions: Whether to add research suggestions
    """
    # Add source citations (for in-tree matches)
    if enable_sources and family_tree:
        enhance_message_with_sources(person, family_tree, format_data)
    else:
        format_data['source_citations'] = ""

    # Add relationship diagram (for in-tree matches)
    if enable_diagrams and family_tree:
        enhance_message_with_relationship_diagram(person, family_tree, format_data)
    else:
        format_data['relationship_diagram'] = ""

    # Add research suggestions (optional, can be verbose)
    if enable_suggestions:
        enhance_message_with_research_suggestions(person, family_tree, format_data)
    else:
        format_data['research_suggestions'] = ""


# === TESTS ===

def test_enhance_message_with_sources():
    """Test source citation enhancement."""
    from unittest.mock import Mock

    person = Mock()
    person.username = "test_user"

    family_tree = Mock()
    family_tree.gedcom_id = "I123"

    format_data = {}

    # Should not crash even if GEDCOM not available
    enhance_message_with_sources(person, family_tree, format_data)
    assert 'source_citations' in format_data

    logger.info("✓ Source citation enhancement test passed")


def test_enhance_message_with_relationship_diagram():
    """Test relationship diagram enhancement."""
    from unittest.mock import Mock

    person = Mock()
    person.username = "test_user"
    person.first_name = "John"

    family_tree = Mock()
    family_tree.relationship_path = '[{"name": "Wayne", "relationship": "self"}, {"name": "John", "relationship": "cousin"}]'

    format_data = {}

    enhance_message_with_relationship_diagram(person, family_tree, format_data)
    assert 'relationship_diagram' in format_data

    logger.info("✓ Relationship diagram enhancement test passed")


def test_enhance_message_with_research_suggestions():
    """Test research suggestion enhancement."""
    from unittest.mock import Mock

    person = Mock()
    person.username = "test_user"
    person.birth_year = 1950

    family_tree = Mock()
    family_tree.common_ancestor_name = "William Gault"

    format_data = {}

    enhance_message_with_research_suggestions(person, family_tree, format_data)
    assert 'research_suggestions' in format_data

    logger.info("✓ Research suggestion enhancement test passed")


def test_enhance_message_format_data_phase5():
    """Test complete Phase 5 enhancement."""
    from unittest.mock import Mock

    person = Mock()
    person.username = "test_user"
    person.first_name = "John"
    person.birth_year = 1950

    family_tree = Mock()
    family_tree.gedcom_id = "I123"
    family_tree.relationship_path = '[]'
    family_tree.common_ancestor_name = "William Gault"

    dna_match = Mock()

    format_data = {}

    enhance_message_format_data_phase5(
        person, family_tree, dna_match, format_data,
        enable_sources=True,
        enable_diagrams=True,
        enable_suggestions=True
    )

    assert 'source_citations' in format_data
    assert 'relationship_diagram' in format_data
    assert 'research_suggestions' in format_data

    logger.info("✓ Complete Phase 5 enhancement test passed")


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 8 Phase 5 Integration Tests", __name__)
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Source Citation Enhancement",
            test_enhance_message_with_sources,
            "Source citations added to format data",
            "Test source citation enhancement",
            "Verifying source citation integration"
        )

        suite.run_test(
            "Relationship Diagram Enhancement",
            test_enhance_message_with_relationship_diagram,
            "Relationship diagrams added to format data",
            "Test relationship diagram enhancement",
            "Verifying relationship diagram integration"
        )

        suite.run_test(
            "Research Suggestion Enhancement",
            test_enhance_message_with_research_suggestions,
            "Research suggestions added to format data",
            "Test research suggestion enhancement",
            "Verifying research suggestion integration"
        )

        suite.run_test(
            "Complete Phase 5 Enhancement",
            test_enhance_message_format_data_phase5,
            "All Phase 5 features integrated",
            "Test complete Phase 5 enhancement",
            "Verifying all Phase 5 features work together"
        )

    return suite.finish_suite()


if __name__ == "__main__":
    print("🤖 Running Action 8 Phase 5 Integration Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

