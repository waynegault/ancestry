"""
Research Guidance AI Prompts

Phase 5.6: Research Guidance AI Prompt
Provides AI prompts for generating personalized genealogical research guidance.

This module creates structured prompts that can be used with AI models to generate
helpful, contextual research suggestions for DNA matches.
"""

from standard_imports import *

logger = logging.getLogger(__name__)


def create_research_guidance_prompt(
    person_name: str,
    relationship: Optional[str] = None,
    shared_dna_cm: Optional[float] = None,
    common_ancestors: Optional[list[str]] = None,
    missing_info: Optional[list[str]] = None,
    available_records: Optional[list[dict[str, Any]]] = None
) -> str:
    """
    Create an AI prompt for generating research guidance.

    Phase 5.6: Research Guidance AI Prompt
    Creates structured prompts for AI-powered research suggestions.

    Args:
        person_name: Name of the person to research
        relationship: Predicted relationship (e.g., "2nd cousin")
        shared_dna_cm: Amount of shared DNA in centiMorgans
        common_ancestors: List of common ancestor names
        missing_info: List of missing information (e.g., ["birth date", "death place"])
        available_records: List of available records with type and details

    Returns:
        Formatted AI prompt string
    """
    prompt_parts = []

    # Header
    prompt_parts.append("Generate genealogical research guidance for the following person:")
    prompt_parts.append("")

    # Person information
    prompt_parts.append(f"Person: {person_name}")

    if relationship:
        prompt_parts.append(f"Relationship: {relationship}")

    if shared_dna_cm:
        prompt_parts.append(f"Shared DNA: {shared_dna_cm} cM")

    # Common ancestors
    if common_ancestors:
        prompt_parts.append("")
        prompt_parts.append("Common Ancestors:")
        for ancestor in common_ancestors:
            prompt_parts.append(f"  - {ancestor}")

    # Missing information
    if missing_info:
        prompt_parts.append("")
        prompt_parts.append("Missing Information:")
        for info in missing_info:
            prompt_parts.append(f"  - {info}")

    # Available records
    if available_records:
        prompt_parts.append("")
        prompt_parts.append("Available Records:")
        for record in available_records:
            record_type = record.get('type', 'Unknown')
            details = record.get('details', {})
            date = details.get('date', 'Unknown date')
            place = details.get('place', 'Unknown place')
            prompt_parts.append(f"  - {record_type}: {date}, {place}")

    # Request
    prompt_parts.append("")
    prompt_parts.append("Please provide:")
    prompt_parts.append("1. Specific research suggestions to fill in missing information")
    prompt_parts.append("2. Recommended Ancestry collections to search")
    prompt_parts.append("3. Alternative record sources to explore")
    prompt_parts.append("4. Tips for breaking through brick walls")

    return "\n".join(prompt_parts)


def create_conversation_response_prompt(
    person_name: str,
    their_message: str,
    conversation_context: Optional[str] = None,
    relationship_info: Optional[dict[str, Any]] = None
) -> str:
    """
    Create an AI prompt for generating a conversational response.

    Phase 5.6: Research Guidance AI Prompt
    Creates prompts for AI-powered conversation responses.

    Args:
        person_name: Name of the person we're responding to
        their_message: The message they sent
        conversation_context: Previous conversation context
        relationship_info: Dictionary with relationship details

    Returns:
        Formatted AI prompt string
    """
    prompt_parts = []

    # Header
    prompt_parts.append("Generate a helpful, friendly response to this DNA match:")
    prompt_parts.append("")

    # Person information
    prompt_parts.append(f"DNA Match: {person_name}")

    if relationship_info:
        if 'relationship' in relationship_info:
            prompt_parts.append(f"Relationship: {relationship_info['relationship']}")
        if 'shared_dna_cm' in relationship_info:
            prompt_parts.append(f"Shared DNA: {relationship_info['shared_dna_cm']} cM")

    # Their message
    prompt_parts.append("")
    prompt_parts.append("Their Message:")
    prompt_parts.append(f'"{their_message}"')

    # Conversation context
    if conversation_context:
        prompt_parts.append("")
        prompt_parts.append("Previous Conversation:")
        prompt_parts.append(conversation_context)

    # Request
    prompt_parts.append("")
    prompt_parts.append("Please generate a response that:")
    prompt_parts.append("1. Addresses their questions or comments")
    prompt_parts.append("2. Provides helpful genealogical information")
    prompt_parts.append("3. Suggests next steps for collaboration")
    prompt_parts.append("4. Maintains a friendly, professional tone")

    return "\n".join(prompt_parts)


def create_brick_wall_analysis_prompt(
    ancestor_name: str,
    known_facts: list[str],
    unknown_facts: list[str],
    searched_collections: Optional[list[str]] = None
) -> str:
    """
    Create an AI prompt for brick wall analysis.

    Phase 5.6: Research Guidance AI Prompt
    Creates prompts for analyzing genealogical brick walls.

    Args:
        ancestor_name: Name of the ancestor with missing information
        known_facts: List of known facts about the ancestor
        unknown_facts: List of unknown facts we're trying to find
        searched_collections: List of collections already searched

    Returns:
        Formatted AI prompt string
    """
    prompt_parts = []

    # Header
    prompt_parts.append("Analyze this genealogical brick wall and suggest research strategies:")
    prompt_parts.append("")

    # Ancestor information
    prompt_parts.append(f"Ancestor: {ancestor_name}")
    prompt_parts.append("")

    # Known facts
    prompt_parts.append("Known Facts:")
    for fact in known_facts:
        prompt_parts.append(f"  ✓ {fact}")

    # Unknown facts
    prompt_parts.append("")
    prompt_parts.append("Unknown Facts (Research Goals):")
    for fact in unknown_facts:
        prompt_parts.append(f"  ? {fact}")

    # Searched collections
    if searched_collections:
        prompt_parts.append("")
        prompt_parts.append("Already Searched:")
        for collection in searched_collections:
            prompt_parts.append(f"  - {collection}")

    # Request
    prompt_parts.append("")
    prompt_parts.append("Please provide:")
    prompt_parts.append("1. Alternative search strategies")
    prompt_parts.append("2. Overlooked record types to explore")
    prompt_parts.append("3. Collateral research suggestions (siblings, neighbors, etc.)")
    prompt_parts.append("4. DNA testing strategies if applicable")

    return "\n".join(prompt_parts)


# ==============================================
# TESTS
# ==============================================


def test_basic_research_guidance_prompt():
    """Test basic research guidance prompt creation."""
    prompt = create_research_guidance_prompt(
        person_name="Fraser Gault",
        relationship="father",
        shared_dna_cm=3400.0,
        missing_info=["birth certificate", "military service"]
    )

    assert "Fraser Gault" in prompt, "Should include person name"
    assert "father" in prompt, "Should include relationship"
    assert "3400" in prompt, "Should include shared DNA"
    assert "birth certificate" in prompt, "Should include missing info"
    assert "military service" in prompt, "Should include missing info"


def test_prompt_with_common_ancestors():
    """Test prompt with common ancestors."""
    prompt = create_research_guidance_prompt(
        person_name="John Smith",
        common_ancestors=["William Gault", "Mary Brown"]
    )

    assert "John Smith" in prompt, "Should include person name"
    assert "William Gault" in prompt, "Should include common ancestor"
    assert "Mary Brown" in prompt, "Should include common ancestor"


def test_prompt_with_available_records():
    """Test prompt with available records."""
    records = [
        {
            'type': 'Birth',
            'details': {'date': '1941', 'place': 'Banff, Scotland'}
        },
        {
            'type': 'Census',
            'details': {'date': '1951', 'place': 'Aberdeen, Scotland'}
        }
    ]

    prompt = create_research_guidance_prompt(
        person_name="Fraser Gault",
        available_records=records
    )

    assert "Fraser Gault" in prompt, "Should include person name"
    assert "Birth" in prompt, "Should include record type"
    assert "1941" in prompt, "Should include record date"
    assert "Banff" in prompt, "Should include record place"


def test_conversation_response_prompt():
    """Test conversation response prompt creation."""
    prompt = create_conversation_response_prompt(
        person_name="John Smith",
        their_message="Do you have any information about William Gault?",
        relationship_info={'relationship': '2nd cousin', 'shared_dna_cm': 212.0}
    )

    assert "John Smith" in prompt, "Should include person name"
    assert "William Gault" in prompt, "Should include their message"
    assert "2nd cousin" in prompt, "Should include relationship"
    assert "212" in prompt, "Should include shared DNA"


def test_brick_wall_analysis_prompt():
    """Test brick wall analysis prompt creation."""
    prompt = create_brick_wall_analysis_prompt(
        ancestor_name="William Gault",
        known_facts=["Born 1875 in Banff", "Married Mary Brown 1895"],
        unknown_facts=["Death date", "Death place", "Occupation"],
        searched_collections=["Scotland Birth Records", "Scotland Census 1881-1901"]
    )

    assert "William Gault" in prompt, "Should include ancestor name"
    assert "Born 1875" in prompt, "Should include known facts"
    assert "Death date" in prompt, "Should include unknown facts"
    assert "Scotland Birth Records" in prompt, "Should include searched collections"


def research_guidance_prompts_module_tests() -> bool:
    """Run all research guidance prompts tests."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Research Guidance AI Prompts", __name__)
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Basic research guidance prompt",
            test_basic_research_guidance_prompt,
            "Research guidance prompts created correctly with all details",
            "Test basic research guidance prompt",
            "Testing prompt with person info, relationship, DNA, and missing info"
        )

        suite.run_test(
            "Prompt with common ancestors",
            test_prompt_with_common_ancestors,
            "Prompts include common ancestor information",
            "Test prompt with common ancestors",
            "Testing common ancestor inclusion in prompts"
        )

        suite.run_test(
            "Prompt with available records",
            test_prompt_with_available_records,
            "Prompts include available record information",
            "Test prompt with available records",
            "Testing record inclusion in prompts"
        )

        suite.run_test(
            "Conversation response prompt",
            test_conversation_response_prompt,
            "Conversation prompts created correctly",
            "Test conversation response prompt",
            "Testing conversational prompt generation"
        )

        suite.run_test(
            "Brick wall analysis prompt",
            test_brick_wall_analysis_prompt,
            "Brick wall analysis prompts created correctly",
            "Test brick wall analysis prompt",
            "Testing brick wall prompt with known/unknown facts"
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return research_guidance_prompts_module_tests()


if __name__ == "__main__":
    print("🤖 Running Research Guidance AI Prompts comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

