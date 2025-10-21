"""
Phase 5 Integration Tests

Tests the integration of all Phase 5 Research Assistant Features:
- P5.1: Source Citation Extraction
- P5.2: Research Suggestion Generation
- P5.3: Enhanced MS To-Do Task Creation
- P5.4: Relationship Diagram Generation
- P5.5: Record Sharing Capabilities
- P5.6: Research Guidance AI Prompts

This module verifies that all features work together correctly.
"""

from standard_imports import *

logger = logging.getLogger(__name__)


def test_source_citations_integration():
    """Test source citation extraction integration."""
    from gedcom_utils import format_source_citations, get_person_sources

    # This would normally use real GEDCOM data
    # For testing, we verify the functions exist and have correct signatures
    assert callable(get_person_sources), "get_person_sources should be callable"
    assert callable(format_source_citations), "format_source_citations should be callable"

    logger.info("✓ Source citation functions available")


def test_research_suggestions_integration():
    """Test research suggestion generation integration."""
    from research_suggestions import generate_research_suggestions

    common_ancestors = [
        {'name': 'William Gault', 'birth_year': 1875, 'birth_place': 'Banff, Scotland'}
    ]
    locations = ["Banff, Scotland", "Aberdeen, Scotland"]
    time_periods = ["1870s", "1880s"]

    result = generate_research_suggestions(
        common_ancestors=common_ancestors,
        locations=locations,
        time_periods=time_periods
    )

    assert isinstance(result, dict), "Should return dict of suggestions"
    assert 'collections' in result, "Should have collections key"

    logger.info(f"✓ Generated research suggestions with {len(result.get('collections', []))} collections")


def test_enhanced_tasks_integration():
    """Test enhanced MS To-Do task creation integration."""
    from ms_graph_utils import create_todo_task

    # Verify function exists with correct signature
    assert callable(create_todo_task), "create_todo_task should be callable"

    # Check that it accepts the new parameters
    import inspect
    sig = inspect.signature(create_todo_task)
    params = list(sig.parameters.keys())

    assert 'importance' in params, "Should accept importance parameter"
    assert 'due_date' in params, "Should accept due_date parameter"
    assert 'categories' in params, "Should accept categories parameter"

    logger.info("✓ Enhanced task creation function available with new parameters")


def test_relationship_diagrams_integration():
    """Test relationship diagram generation integration."""
    from relationship_diagram import (
        format_relationship_for_message,
        generate_relationship_diagram,
    )

    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'}
    ]

    # Test vertical diagram
    diagram = generate_relationship_diagram("Wayne", "Fraser", path, "vertical")
    assert "Wayne Gault" in diagram, "Should include first person"
    assert "Fraser Gault" in diagram, "Should include second person"

    # Test message formatting
    message = format_relationship_for_message("Wayne", "Fraser", path, include_diagram=True)
    assert "father" in message.lower(), "Should include relationship"

    logger.info("✓ Relationship diagrams generated successfully")


def test_record_sharing_integration():
    """Test record sharing capabilities integration."""
    from record_sharing import (
        create_record_sharing_message,
        format_record_reference,
    )

    details = {
        'date': '1941',
        'place': 'Banff, Scotland',
        'source': 'Birth Certificate'
    }

    # Test single record
    ref = format_record_reference('Birth', 'Fraser Gault', details)
    assert 'Fraser Gault' in ref, "Should include person name"
    assert '1941' in ref, "Should include date"

    # Test complete message
    records = [{'type': 'Birth', 'details': details}]
    message = create_record_sharing_message('Fraser Gault', records, "Found records:")
    assert 'Fraser Gault' in message, "Should include person name"

    logger.info("✓ Record sharing formatted successfully")


def test_ai_prompts_integration():
    """Test research guidance AI prompts integration."""
    from research_guidance_prompts import (
        create_brick_wall_analysis_prompt,
        create_conversation_response_prompt,
        create_research_guidance_prompt,
    )

    # Test research guidance prompt
    prompt1 = create_research_guidance_prompt(
        person_name="Fraser Gault",
        relationship="father",
        missing_info=["birth certificate"]
    )
    assert "Fraser Gault" in prompt1, "Should include person name"
    assert "birth certificate" in prompt1, "Should include missing info"

    # Test conversation prompt
    prompt2 = create_conversation_response_prompt(
        person_name="John Smith",
        their_message="Do you have info about William Gault?"
    )
    assert "John Smith" in prompt2, "Should include person name"
    assert "William Gault" in prompt2, "Should include their message"

    # Test brick wall prompt
    prompt3 = create_brick_wall_analysis_prompt(
        ancestor_name="William Gault",
        known_facts=["Born 1875"],
        unknown_facts=["Death date"]
    )
    assert "William Gault" in prompt3, "Should include ancestor name"
    assert "Born 1875" in prompt3, "Should include known facts"

    logger.info("✓ AI prompts generated successfully")


def test_complete_workflow_integration():
    """Test complete workflow using all Phase 5 features."""
    from record_sharing import format_record_reference
    from relationship_diagram import generate_relationship_diagram
    from research_guidance_prompts import create_conversation_response_prompt
    from research_suggestions import generate_research_suggestions

    # Scenario: Responding to a DNA match about a common ancestor

    # 1. Generate research suggestions
    common_ancestors = [
        {'name': 'William Gault', 'birth_year': 1875, 'birth_place': 'Banff, Scotland'}
    ]
    result = generate_research_suggestions(
        common_ancestors=common_ancestors,
        locations=["Banff, Scotland", "Aberdeen, Scotland"],
        time_periods=["1870s", "1880s"]
    )
    assert 'collections' in result, "Should generate suggestions"

    # 2. Format a record reference
    record_ref = format_record_reference(
        'Birth',
        'William Gault',
        {'date': '1875', 'place': 'Banff, Scotland', 'source': 'Birth Certificate'}
    )
    assert 'William Gault' in record_ref, "Should format record"

    # 3. Generate a relationship diagram
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'William Gault', 'relationship': 'grandfather'}
    ]
    diagram = generate_relationship_diagram("Wayne", "William", path, "vertical")
    assert 'William Gault' in diagram, "Should generate diagram"

    # 4. Create an AI prompt for response
    prompt = create_conversation_response_prompt(
        person_name="DNA Match",
        their_message="Do you have info about William Gault?",
        relationship_info={'relationship': '3rd cousin', 'shared_dna_cm': 98.0}
    )
    assert 'William Gault' in prompt, "Should create prompt"

    logger.info("✓ Complete workflow executed successfully")
    logger.info(f"  - Generated {len(result.get('collections', []))} research collection suggestions")
    logger.info("  - Formatted record reference")
    logger.info("  - Created relationship diagram")
    logger.info("  - Generated AI prompt")


def phase5_integration_tests() -> bool:
    """Run all Phase 5 integration tests."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Phase 5 Integration Tests", __name__)
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Source Citations Integration",
            test_source_citations_integration,
            "Source citation functions available and working",
            "Test source citation integration",
            "Verifying source citation extraction functions"
        )

        suite.run_test(
            "Research Suggestions Integration",
            test_research_suggestions_integration,
            "Research suggestions generated correctly",
            "Test research suggestion integration",
            "Verifying research suggestion generation"
        )

        suite.run_test(
            "Enhanced Tasks Integration",
            test_enhanced_tasks_integration,
            "Enhanced task creation functions available",
            "Test enhanced task integration",
            "Verifying enhanced MS To-Do task creation"
        )

        suite.run_test(
            "Relationship Diagrams Integration",
            test_relationship_diagrams_integration,
            "Relationship diagrams generated correctly",
            "Test relationship diagram integration",
            "Verifying relationship diagram generation"
        )

        suite.run_test(
            "Record Sharing Integration",
            test_record_sharing_integration,
            "Record sharing formatted correctly",
            "Test record sharing integration",
            "Verifying record sharing capabilities"
        )

        suite.run_test(
            "AI Prompts Integration",
            test_ai_prompts_integration,
            "AI prompts generated correctly",
            "Test AI prompt integration",
            "Verifying research guidance AI prompts"
        )

        suite.run_test(
            "Complete Workflow Integration",
            test_complete_workflow_integration,
            "Complete workflow executed successfully",
            "Test complete workflow",
            "Verifying all Phase 5 features work together"
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return phase5_integration_tests()


if __name__ == "__main__":
    print("🤖 Running Phase 5 Integration Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

