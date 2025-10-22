"""
Relationship Diagram Generation

Phase 5.4: Relationship Diagram Generation
Creates ASCII/text-based relationship diagrams showing the path between two people.

This module generates visual representations of genealogical relationships
for use in messages and responses.
"""

from standard_imports import *

logger = logging.getLogger(__name__)


def generate_relationship_diagram(
    person1_name: str,
    person2_name: str,
    relationship_path: list[dict[str, str]],
    diagram_style: str = "vertical"
) -> str:
    """
    Generate a text-based relationship diagram showing the path between two people.

    Phase 5.4: Relationship Diagram Generation
    Creates ASCII diagrams for genealogical relationships.

    Args:
        person1_name: Name of the first person (usually the tree owner)
        person2_name: Name of the second person (usually the DNA match)
        relationship_path: List of relationship steps, each with 'name' and 'relationship' keys
        diagram_style: Style of diagram ('vertical', 'horizontal', 'compact')

    Returns:
        String containing the ASCII diagram

    Example relationship_path:
        [
            {'name': 'Wayne Gault', 'relationship': 'self'},
            {'name': 'Fraser Gault', 'relationship': 'father'},
            {'name': 'John Gault', 'relationship': 'grandfather'},
            {'name': 'Mary Smith', 'relationship': 'grandmother'},
            {'name': 'Jane Smith', 'relationship': 'aunt'},
            {'name': 'Bob Smith', 'relationship': '1st cousin'}
        ]
    """
    if not relationship_path or len(relationship_path) < 2:
        return f"{person1_name} → {person2_name}"

    if diagram_style == "vertical":
        return _generate_vertical_diagram(person1_name, person2_name, relationship_path)
    if diagram_style == "horizontal":
        return _generate_horizontal_diagram(person1_name, person2_name, relationship_path)
    if diagram_style == "compact":
        return _generate_compact_diagram(person1_name, person2_name, relationship_path)
    return _generate_vertical_diagram(person1_name, person2_name, relationship_path)


def _generate_vertical_diagram(
    _: str,
    __: str,
    relationship_path: list[dict[str, str]]
) -> str:
    """Generate a vertical relationship diagram."""
    lines = []
    lines.append("Relationship Path:")
    lines.append("=" * 50)
    lines.append("")

    for i, step in enumerate(relationship_path):
        name = step.get('name', 'Unknown')

        # Add the person
        if i == 0:
            lines.append(f"  {name} (You)")
        else:
            lines.append(f"  {name}")

        # Add the relationship arrow (except for the last person)
        if i < len(relationship_path) - 1:
            next_relationship = relationship_path[i + 1].get('relationship', '')
            lines.append(f"    ↓ ({next_relationship})")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


def _generate_horizontal_diagram(
    _: str,
    __: str,
    relationship_path: list[dict[str, str]]
) -> str:
    """Generate a horizontal relationship diagram."""
    names = [step.get('name', 'Unknown') for step in relationship_path]
    relationships = [step.get('relationship', '') for step in relationship_path[1:]]

    # Build the diagram
    diagram_parts = []
    for i, name in enumerate(names):
        if i == 0:
            diagram_parts.append(f"{name} (You)")
        else:
            diagram_parts.append(name)

        if i < len(names) - 1:
            rel = relationships[i] if i < len(relationships) else ''
            diagram_parts.append(f" → ({rel}) → ")

    return "".join(diagram_parts)


def _generate_compact_diagram(
    person1_name: str,
    person2_name: str,
    relationship_path: list[dict[str, str]]
) -> str:
    """Generate a compact relationship diagram."""
    if len(relationship_path) <= 2:
        return f"{person1_name} → {person2_name}"

    # Show first person, middle indicator, and last person
    first = relationship_path[0].get('name', 'Unknown')
    last = relationship_path[-1].get('name', 'Unknown')
    middle_count = len(relationship_path) - 2

    if middle_count == 0:
        return f"{first} → {last}"
    if middle_count == 1:
        middle = relationship_path[1].get('name', 'Unknown')
        return f"{first} → {middle} → {last}"
    return f"{first} → ... ({middle_count} generations) ... → {last}"


def format_relationship_for_message(
    person1_name: str,
    person2_name: str,
    relationship_path: list[dict[str, str]],
    include_diagram: bool = True
) -> str:
    """
    Format a relationship path for inclusion in a message.

    Phase 5.4: Relationship Diagram Generation
    Formats relationship information for messages with optional diagram.

    Args:
        person1_name: Name of the first person
        person2_name: Name of the second person
        relationship_path: List of relationship steps
        include_diagram: Whether to include the visual diagram

    Returns:
        Formatted string for message inclusion
    """
    if not relationship_path:
        return f"I believe you are related to {person2_name}, but I haven't determined the exact path yet."

    # Generate summary
    if len(relationship_path) == 2:
        relationship = relationship_path[1].get('relationship', 'relative')
        summary = f"You are {person2_name}'s {relationship}."
    else:
        summary = f"You are connected to {person2_name} through {len(relationship_path) - 1} generations."

    if not include_diagram:
        return summary

    # Add diagram
    diagram = generate_relationship_diagram(person1_name, person2_name, relationship_path, "vertical")

    return f"{summary}\n\n{diagram}"


# ==============================================
# TESTS
# ==============================================


def test_vertical_diagram():
    """Test vertical diagram generation."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'John Gault', 'relationship': 'grandfather'}
    ]

    diagram = generate_relationship_diagram("Wayne Gault", "John Gault", path, "vertical")

    assert "Wayne Gault (You)" in diagram, "Should include first person with (You)"
    assert "Fraser Gault" in diagram, "Should include middle person"
    assert "John Gault" in diagram, "Should include last person"
    assert "father" in diagram, "Should include relationship"
    assert "grandfather" in diagram, "Should include relationship"
    assert "↓" in diagram, "Should include arrow"


def test_horizontal_diagram():
    """Test horizontal diagram generation."""
    path = [
        {'name': 'Wayne', 'relationship': 'self'},
        {'name': 'Fraser', 'relationship': 'father'},
        {'name': 'John', 'relationship': 'grandfather'}
    ]

    diagram = generate_relationship_diagram("Wayne", "John", path, "horizontal")

    assert "Wayne (You)" in diagram, "Should include first person"
    assert "Fraser" in diagram, "Should include middle person"
    assert "John" in diagram, "Should include last person"
    assert "→" in diagram, "Should include arrows"


def test_compact_diagram():
    """Test compact diagram generation."""
    # Test with 2 people
    path2 = [
        {'name': 'Wayne', 'relationship': 'self'},
        {'name': 'Fraser', 'relationship': 'father'}
    ]
    diagram2 = generate_relationship_diagram("Wayne", "Fraser", path2, "compact")
    assert "Wayne → Fraser" in diagram2, "Should show direct connection"

    # Test with many people
    path_many = [
        {'name': 'Wayne', 'relationship': 'self'},
        {'name': 'Person2', 'relationship': 'rel2'},
        {'name': 'Person3', 'relationship': 'rel3'},
        {'name': 'Person4', 'relationship': 'rel4'},
        {'name': 'Person5', 'relationship': 'rel5'}
    ]
    diagram_many = generate_relationship_diagram("Wayne", "Person5", path_many, "compact")
    assert "generations" in diagram_many, "Should indicate multiple generations"


def test_message_formatting():
    """Test relationship formatting for messages."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'}
    ]

    # Test with diagram
    message_with_diagram = format_relationship_for_message("Wayne Gault", "Fraser Gault", path, include_diagram=True)
    assert "father" in message_with_diagram, "Should include relationship"
    assert "Relationship Path:" in message_with_diagram, "Should include diagram"

    # Test without diagram
    message_without_diagram = format_relationship_for_message("Wayne Gault", "Fraser Gault", path, include_diagram=False)
    assert "father" in message_without_diagram, "Should include relationship"
    assert "Relationship Path:" not in message_without_diagram, "Should not include diagram"


def _test_empty_relationship_path():
    """Test diagram generation with empty relationship path."""
    diagram = generate_relationship_diagram(
        "Wayne Gault",
        "John Smith",
        [],
        "vertical"
    )

    assert "Wayne Gault" in diagram, "Should include person1 name"
    assert "John Smith" in diagram, "Should include person2 name"
    assert "→" in diagram, "Should have simple arrow for empty path"

    logger.info("✓ Empty relationship path handled correctly")
    return True


def _test_single_step_relationship():
    """Test diagram generation with single step relationship."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'}
    ]

    diagram = generate_relationship_diagram(
        "Wayne Gault",
        "Fraser Gault",
        path,
        "vertical"
    )

    assert "Wayne Gault" in diagram, "Should include person1 name"
    assert "Fraser Gault" in diagram, "Should include person2 name"
    assert len(diagram) > 10, "Should have formatted diagram"

    logger.info("✓ Single step relationship diagram generated correctly")
    return True


def _test_invalid_diagram_style():
    """Test diagram generation with invalid style defaults to vertical."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'}
    ]

    diagram = generate_relationship_diagram(
        "Wayne Gault",
        "Fraser Gault",
        path,
        "invalid_style"
    )

    # Should default to vertical style
    assert "Wayne Gault" in diagram, "Should include person1 name"
    assert "Fraser Gault" in diagram, "Should include person2 name"
    assert len(diagram) > 10, "Should have formatted diagram"

    logger.info("✓ Invalid diagram style defaults to vertical correctly")
    return True


def _test_all_diagram_styles():
    """Test all three diagram styles produce different output."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'John Gault', 'relationship': 'grandfather'}
    ]

    vertical = generate_relationship_diagram("Wayne Gault", "John Gault", path, "vertical")
    horizontal = generate_relationship_diagram("Wayne Gault", "John Gault", path, "horizontal")
    compact = generate_relationship_diagram("Wayne Gault", "John Gault", path, "compact")

    # All should contain the names
    for diagram in [vertical, horizontal, compact]:
        assert "Wayne Gault" in diagram, "Should include person1 name"
        assert "John Gault" in diagram, "Should include person2 name"

    # They should be different (different formatting)
    assert vertical != horizontal, "Vertical and horizontal should differ"
    assert vertical != compact, "Vertical and compact should differ"
    assert horizontal != compact, "Horizontal and compact should differ"

    logger.info("✓ All diagram styles produce different output correctly")
    return True


def _test_format_relationship_simple():
    """Test simple relationship formatting for messages."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'}
    ]

    formatted = format_relationship_for_message("Wayne Gault", "Fraser Gault", path)

    assert isinstance(formatted, str), "Should return string"
    assert "Wayne Gault" in formatted, "Should include person1 name"
    assert "Fraser Gault" in formatted, "Should include person2 name"
    assert "father" in formatted.lower(), "Should mention relationship"

    logger.info("✓ Simple relationship formatted correctly for message")
    return True


def _test_format_relationship_complex():
    """Test complex relationship formatting for messages."""
    path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'John Gault', 'relationship': 'grandfather'},
        {'name': 'Mary Smith', 'relationship': 'grandmother'},
        {'name': 'Jane Smith', 'relationship': 'aunt'}
    ]

    formatted = format_relationship_for_message("Wayne Gault", "Jane Smith", path)

    assert isinstance(formatted, str), "Should return string"
    assert "Wayne Gault" in formatted, "Should include person1 name"
    assert "Jane Smith" in formatted, "Should include person2 name"
    assert len(formatted) > 20, "Should have detailed description"

    logger.info("✓ Complex relationship formatted correctly for message")
    return True


def _test_format_relationship_empty_path():
    """Test relationship formatting with empty path."""
    formatted = format_relationship_for_message("Wayne Gault", "John Smith", [])

    assert isinstance(formatted, str), "Should return string"
    assert "John Smith" in formatted, "Should include person2 name"
    assert "related" in formatted.lower(), "Should mention relationship"
    assert "haven't determined" in formatted or "not determined" in formatted, "Should indicate path unknown"

    logger.info("✓ Empty path relationship formatted correctly")
    return True


def relationship_diagram_module_tests() -> bool:
    """Run all relationship diagram tests."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Relationship Diagram Generation", __name__)
    suite.start_suite()

    with suppress_logging():
        # Edge case tests
        suite.run_test(
            "Empty relationship path",
            _test_empty_relationship_path,
            "Empty relationship path handled correctly",
            "Test diagram generation with empty relationship path",
            "Verify simple arrow format for empty path"
        )

        suite.run_test(
            "Single step relationship",
            _test_single_step_relationship,
            "Single step relationship diagram generated correctly",
            "Test diagram generation with single step relationship",
            "Verify diagram with just two people (self and one relation)"
        )

        suite.run_test(
            "Invalid diagram style",
            _test_invalid_diagram_style,
            "Invalid diagram style defaults to vertical correctly",
            "Test diagram generation with invalid style",
            "Verify invalid style defaults to vertical format"
        )

        # Style comparison test
        suite.run_test(
            "All diagram styles",
            _test_all_diagram_styles,
            "All diagram styles produce different output correctly",
            "Test all three diagram styles produce different output",
            "Verify vertical, horizontal, and compact styles are distinct"
        )

        # Original tests
        suite.run_test(
            "Vertical diagram generation",
            test_vertical_diagram,
            "Vertical diagrams generated correctly with proper formatting and arrows",
            "Test vertical relationship diagram generation",
            "Testing vertical diagram with multiple generations and relationship labels"
        )

        suite.run_test(
            "Horizontal diagram generation",
            test_horizontal_diagram,
            "Horizontal diagrams generated correctly with proper formatting and arrows",
            "Test horizontal relationship diagram generation",
            "Testing horizontal diagram with inline relationship display"
        )

        suite.run_test(
            "Compact diagram generation",
            test_compact_diagram,
            "Compact diagrams generated correctly for various path lengths",
            "Test compact relationship diagram generation",
            "Testing compact diagram with generation count for long paths"
        )

        # Message formatting tests
        suite.run_test(
            "Simple relationship formatting",
            _test_format_relationship_simple,
            "Simple relationship formatted correctly for message",
            "Test simple relationship formatting for messages",
            "Verify formatting with single relationship step"
        )

        suite.run_test(
            "Complex relationship formatting",
            _test_format_relationship_complex,
            "Complex relationship formatted correctly for message",
            "Test complex relationship formatting for messages",
            "Verify formatting with multiple relationship steps"
        )

        suite.run_test(
            "Empty path relationship formatting",
            _test_format_relationship_empty_path,
            "Empty path relationship formatted correctly",
            "Test relationship formatting with empty path",
            "Verify formatting handles empty path gracefully"
        )

        suite.run_test(
            "Message formatting with relationships",
            test_message_formatting,
            "Relationship information formatted correctly for message inclusion",
            "Test relationship formatting for messages",
            "Testing message formatting with and without diagrams"
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return relationship_diagram_module_tests()


if __name__ == "__main__":
    print("🤖 Running Relationship Diagram Generation comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

