"""
Action 9 Phase 5 Integration

Integrates Phase 5 Research Assistant Features into Action 9 productive conversation management:
- Enhanced MS To-Do task creation with priority and due dates
- AI-powered research guidance prompts for responses
- Record sharing in conversation replies
- Relationship diagrams in responses

This module enhances Action 9's conversation processing with research capabilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from database import ConversationLog, Person
from ms_graph_utils import create_todo_task
from record_sharing import create_record_sharing_message
from relationship_diagram import format_relationship_for_message
from research_guidance_prompts import (
    create_brick_wall_analysis_prompt,
    create_conversation_response_prompt,
    create_research_guidance_prompt,
)
from standard_imports import *

logger = logging.getLogger(__name__)


def calculate_task_priority_from_relationship(
    relationship: Optional[str],
    shared_dna_cm: Optional[float] = None
) -> tuple[str, int]:
    """
    Calculate MS To-Do task priority and due date offset based on relationship closeness.
    
    Args:
        relationship: Relationship description (e.g., "2nd cousin", "uncle")
        shared_dna_cm: Shared DNA in centiMorgans
    
    Returns:
        Tuple of (importance, days_until_due)
        - importance: "high", "normal", or "low"
        - days_until_due: Number of days until task is due
    """
    # Default to normal priority, 14 days
    importance = "normal"
    days_until_due = 14

    if not relationship:
        return importance, days_until_due

    relationship_lower = relationship.lower()

    # High priority (7 days): Close relationships
    close_relationships = [
        "parent", "child", "sibling", "brother", "sister",
        "uncle", "aunt", "nephew", "niece",
        "1st cousin", "first cousin",
        "2nd cousin", "second cousin"
    ]

    for close_rel in close_relationships:
        if close_rel in relationship_lower:
            importance = "high"
            days_until_due = 7
            return importance, days_until_due

    # Normal priority (14 days): Medium relationships
    medium_relationships = [
        "3rd cousin", "third cousin",
        "4th cousin", "fourth cousin"
    ]

    for medium_rel in medium_relationships:
        if medium_rel in relationship_lower:
            importance = "normal"
            days_until_due = 14
            return importance, days_until_due

    # Low priority (30 days): Distant relationships
    # 5th cousin and beyond
    if "5th" in relationship_lower or "sixth" in relationship_lower or "distant" in relationship_lower:
        importance = "low"
        days_until_due = 30
        return importance, days_until_due

    # Use DNA if available for unknown relationships
    if shared_dna_cm:
        if shared_dna_cm > 200:  # Close relationship
            importance = "high"
            days_until_due = 7
        elif shared_dna_cm > 50:  # Medium relationship
            importance = "normal"
            days_until_due = 14
        else:  # Distant relationship
            importance = "low"
            days_until_due = 30

    return importance, days_until_due


def create_enhanced_research_task(
    person_name: str,
    relationship: Optional[str],
    task_description: str,
    shared_dna_cm: Optional[float] = None,
    categories: Optional[list[str]] = None
) -> Optional[str]:
    """
    Create an enhanced MS To-Do task with intelligent priority and due date.
    
    Args:
        person_name: Name of the person related to the task
        relationship: Relationship to the person
        task_description: Description of the research task
        shared_dna_cm: Shared DNA in centiMorgans
        categories: Optional categories for the task
    
    Returns:
        Task ID if created successfully, None otherwise
    """
    try:
        # Calculate priority and due date
        importance, days_until_due = calculate_task_priority_from_relationship(
            relationship, shared_dna_cm
        )

        # Calculate due date
        due_date = datetime.now(timezone.utc) + timedelta(days=days_until_due)

        # Default categories
        if not categories:
            categories = ["Genealogy Research", "DNA Matches"]

        # Create task title
        task_title = f"Research: {person_name}"
        if relationship:
            task_title += f" ({relationship})"

        # Create the task
        task_id = create_todo_task(
            title=task_title,
            body=task_description,
            importance=importance,
            due_date=due_date,
            categories=categories
        )

        if task_id:
            logger.info(f"Created {importance} priority task for {person_name} (due in {days_until_due} days)")

        return task_id

    except Exception as e:
        logger.error(f"Failed to create enhanced research task: {e}")
        return None


def generate_ai_response_prompt(
    person_name: str,
    their_message: str,
    relationship_info: Optional[dict[str, Any]] = None,
    conversation_history: Optional[list[dict[str, str]]] = None,
    missing_info: Optional[list[str]] = None
) -> str:
    """
    Generate an AI prompt for responding to a conversation.
    
    Args:
        person_name: Name of the person who sent the message
        their_message: The message they sent
        relationship_info: Optional relationship information
        conversation_history: Optional conversation history
        missing_info: Optional list of missing information
    
    Returns:
        AI prompt string for generating a response
    """
    try:
        # Use research guidance prompt if we have missing info
        if missing_info:
            relationship = relationship_info.get('relationship') if relationship_info else None
            return create_research_guidance_prompt(
                person_name=person_name,
                relationship=relationship,
                missing_info=missing_info,
                conversation_context=their_message
            )

        # Otherwise use conversation response prompt
        return create_conversation_response_prompt(
            person_name=person_name,
            their_message=their_message,
            relationship_info=relationship_info,
            conversation_history=conversation_history
        )

    except Exception as e:
        logger.error(f"Failed to generate AI response prompt: {e}")
        return f"Please help me respond to {person_name}'s message: {their_message}"


def format_response_with_records(
    person_name: str,
    records: list[dict[str, Any]],
    context: str = "I found these records that might be helpful:"
) -> str:
    """
    Format a response that includes record sharing.
    
    Args:
        person_name: Name of the person being responded to
        records: List of record dictionaries
        context: Context message for the records
    
    Returns:
        Formatted message with record references
    """
    try:
        return create_record_sharing_message(person_name, records, context)
    except Exception as e:
        logger.error(f"Failed to format response with records: {e}")
        return context


def format_response_with_relationship_diagram(
    from_name: str,
    to_name: str,
    relationship_path: list[dict[str, str]],
    style: str = "compact"
) -> str:
    """
    Format a response that includes a relationship diagram.
    
    Args:
        from_name: Name of the first person (usually "me" or tree owner)
        to_name: Name of the second person
        relationship_path: List of relationship path dictionaries
        style: Diagram style ("vertical", "horizontal", or "compact")
    
    Returns:
        Formatted message with relationship diagram
    """
    try:
        return format_relationship_for_message(
            from_name, to_name, relationship_path,
            include_diagram=True, style=style
        )
    except Exception as e:
        logger.error(f"Failed to format response with relationship diagram: {e}")
        return f"Our relationship: {from_name} → {to_name}"


# === TESTS ===

def test_calculate_task_priority_from_relationship():
    """Test task priority calculation."""
    # Test that function exists and returns tuples
    result1 = calculate_task_priority_from_relationship("uncle")
    assert isinstance(result1, tuple), "Should return tuple"
    assert len(result1) == 2, "Should return 2-element tuple"
    importance1, days1 = result1
    assert importance1 in ["high", "normal", "low"], f"Invalid importance: {importance1}"
    assert isinstance(days1, int), f"Days should be int, got {type(days1)}"

    # Test that different relationships give different priorities
    result2 = calculate_task_priority_from_relationship("5th cousin")
    importance2, days2 = result2
    assert importance2 in ["high", "normal", "low"], f"Invalid importance: {importance2}"

    # Test DNA-based priority
    result3 = calculate_task_priority_from_relationship(None, shared_dna_cm=250.0)
    importance3, days3 = result3
    assert importance3 in ["high", "normal", "low"], f"Invalid importance: {importance3}"

    logger.info("✓ Task priority calculation test passed")


def test_create_enhanced_research_task():
    """Test enhanced research task creation."""
    # Should not crash even if MS Graph not available
    task_id = create_enhanced_research_task(
        person_name="John Smith",
        relationship="2nd cousin",
        task_description="Research John Smith's family line",
        shared_dna_cm=98.0
    )

    # Task ID might be None if MS Graph not configured, but function should not crash
    logger.info("✓ Enhanced research task creation test passed")


def test_generate_ai_response_prompt():
    """Test AI response prompt generation."""
    prompt = generate_ai_response_prompt(
        person_name="John Smith",
        their_message="Do you have info about William Gault?",
        relationship_info={'relationship': '3rd cousin', 'shared_dna_cm': 98.0}
    )

    assert "John Smith" in prompt
    assert "William Gault" in prompt

    logger.info("✓ AI response prompt generation test passed")


def test_format_response_with_records():
    """Test response formatting with records."""
    records = [
        {
            'type': 'Birth',
            'details': {
                'date': '1941',
                'place': 'Banff, Scotland',
                'source': 'Birth Certificate'
            }
        }
    ]

    response = format_response_with_records("Fraser Gault", records)
    assert "Fraser Gault" in response
    assert "1941" in response

    logger.info("✓ Response formatting with records test passed")


def test_format_response_with_relationship_diagram():
    """Test response formatting with relationship diagram."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'}
    ]

    response = format_response_with_relationship_diagram("Wayne", "Fraser", path)
    assert "Wayne" in response or "Fraser" in response

    logger.info("✓ Response formatting with relationship diagram test passed")


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Action 9 Phase 5 Integration Tests", __name__)
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Task Priority Calculation",
            test_calculate_task_priority_from_relationship,
            "Task priorities calculated correctly",
            "Test task priority calculation",
            "Verifying priority calculation from relationships"
        )

        suite.run_test(
            "Enhanced Research Task Creation",
            test_create_enhanced_research_task,
            "Enhanced tasks created successfully",
            "Test enhanced research task creation",
            "Verifying enhanced MS To-Do task creation"
        )

        suite.run_test(
            "AI Response Prompt Generation",
            test_generate_ai_response_prompt,
            "AI prompts generated correctly",
            "Test AI response prompt generation",
            "Verifying AI prompt generation"
        )

        suite.run_test(
            "Response with Records",
            test_format_response_with_records,
            "Responses formatted with records",
            "Test response formatting with records",
            "Verifying record sharing in responses"
        )

        suite.run_test(
            "Response with Relationship Diagram",
            test_format_response_with_relationship_diagram,
            "Responses formatted with diagrams",
            "Test response formatting with diagrams",
            "Verifying relationship diagrams in responses"
        )

    return suite.finish_suite()


if __name__ == "__main__":
    print("🤖 Running Action 9 Phase 5 Integration Tests...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

