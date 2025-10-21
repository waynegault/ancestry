"""
Demonstration: Research Guidance AI Prompts

Phase 5.6: Research Guidance AI Prompt
Demonstrates the creation of AI prompts for genealogical research guidance.

This script shows how structured prompts can be generated for use with AI models
to provide personalized research suggestions and conversational responses.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging

from research_guidance_prompts import (
    create_brick_wall_analysis_prompt,
    create_conversation_response_prompt,
    create_research_guidance_prompt,
)
from standard_imports import *

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def demonstrate_basic_research_prompt():
    """Demonstrate basic research guidance prompt."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 1: Basic Research Guidance Prompt")
    logger.info("="*80)

    prompt = create_research_guidance_prompt(
        person_name="Fraser Gault",
        relationship="father",
        shared_dna_cm=3400.0,
        missing_info=["birth certificate", "military service records", "death certificate"]
    )

    logger.info("\nGenerated AI Prompt:\n")
    logger.info("-" * 80)
    logger.info(prompt)
    logger.info("-" * 80)
    logger.info("\n" + "="*80)


def demonstrate_prompt_with_ancestors():
    """Demonstrate prompt with common ancestors."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 2: Research Prompt with Common Ancestors")
    logger.info("="*80)

    prompt = create_research_guidance_prompt(
        person_name="John Smith",
        relationship="2nd cousin",
        shared_dna_cm=212.0,
        common_ancestors=["William Gault (1875-1945)", "Mary Brown (1880-1950)"],
        missing_info=["marriage record", "immigration record"]
    )

    logger.info("\nGenerated AI Prompt:\n")
    logger.info("-" * 80)
    logger.info(prompt)
    logger.info("-" * 80)
    logger.info("\n" + "="*80)


def demonstrate_prompt_with_records():
    """Demonstrate prompt with available records."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 3: Research Prompt with Available Records")
    logger.info("="*80)

    records = [
        {
            'type': 'Birth',
            'details': {'date': '1941', 'place': 'Banff, Banffshire, Scotland'}
        },
        {
            'type': 'Census',
            'details': {'date': '1951', 'place': 'Aberdeen, Aberdeenshire, Scotland'}
        },
        {
            'type': 'Marriage',
            'details': {'date': '1965', 'place': 'Edinburgh, Midlothian, Scotland'}
        }
    ]

    prompt = create_research_guidance_prompt(
        person_name="Fraser Gault",
        relationship="father",
        shared_dna_cm=3400.0,
        available_records=records,
        missing_info=["death record", "military service"]
    )

    logger.info("\nGenerated AI Prompt:\n")
    logger.info("-" * 80)
    logger.info(prompt)
    logger.info("-" * 80)
    logger.info("\n" + "="*80)


def demonstrate_conversation_prompt():
    """Demonstrate conversation response prompt."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 4: Conversation Response Prompt")
    logger.info("="*80)

    prompt = create_conversation_response_prompt(
        person_name="Jane Doe",
        their_message="Hi! I see we share DNA. Do you have any information about William Gault who was born in Banff around 1875?",
        conversation_context="Previous message: I mentioned that William Gault is my great-great-grandfather.",
        relationship_info={
            'relationship': '3rd cousin',
            'shared_dna_cm': 98.0
        }
    )

    logger.info("\nGenerated AI Prompt:\n")
    logger.info("-" * 80)
    logger.info(prompt)
    logger.info("-" * 80)
    logger.info("\n" + "="*80)


def demonstrate_brick_wall_prompt():
    """Demonstrate brick wall analysis prompt."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 5: Brick Wall Analysis Prompt")
    logger.info("="*80)

    prompt = create_brick_wall_analysis_prompt(
        ancestor_name="William Gault",
        known_facts=[
            "Born 1875 in Banff, Banffshire, Scotland",
            "Married Mary Brown in 1895 in Aberdeen",
            "Had 5 children between 1896-1910",
            "Lived in Aberdeen in 1901 census"
        ],
        unknown_facts=[
            "Death date and place",
            "Occupation",
            "Parents' names",
            "Siblings"
        ],
        searched_collections=[
            "Scotland Birth Records 1855-1875",
            "Scotland Census 1881-1911",
            "Scotland Marriage Records 1855-1900"
        ]
    )

    logger.info("\nGenerated AI Prompt:\n")
    logger.info("-" * 80)
    logger.info(prompt)
    logger.info("-" * 80)
    logger.info("\n" + "="*80)


def demonstrate_real_world_scenario():
    """Demonstrate real-world research scenario."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 6: Real-World Research Scenario")
    logger.info("="*80)

    logger.info("\nScenario: DNA match asks about a common ancestor\n")
    logger.info("Their message: 'I noticed we share DNA and both have John Gault in our trees.")
    logger.info("               My John was born around 1850 in Scotland. Is this the same person?'\n")

    # Create a comprehensive prompt
    prompt = create_conversation_response_prompt(
        person_name="Robert Johnson",
        their_message="I noticed we share DNA and both have John Gault in our trees. My John was born around 1850 in Scotland. Is this the same person?",
        conversation_context=None,  # First message
        relationship_info={
            'relationship': '4th cousin',
            'shared_dna_cm': 45.0
        }
    )

    logger.info("\nGenerated AI Prompt for Response:\n")
    logger.info("-" * 80)
    logger.info(prompt)
    logger.info("-" * 80)

    logger.info("\n\nThis prompt would be sent to an AI model to generate a helpful response")
    logger.info("that addresses their question and provides relevant genealogical information.")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    """Run all demonstrations."""
    logger.info("\n" + "🎯 "*40)
    logger.info("RESEARCH GUIDANCE AI PROMPTS DEMONSTRATION")
    logger.info("Phase 5.6: AI-Powered Genealogical Research Assistance")
    logger.info("🎯 "*40)

    try:
        # Run demonstrations
        demonstrate_basic_research_prompt()
        demonstrate_prompt_with_ancestors()
        demonstrate_prompt_with_records()
        demonstrate_conversation_prompt()
        demonstrate_brick_wall_prompt()
        demonstrate_real_world_scenario()

        logger.info("\n" + "🎉 "*40)
        logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("\nThese prompts can be used with AI models (like GPT-4, Claude, etc.)")
        logger.info("to generate personalized research guidance and conversational responses.")
        logger.info("🎉 "*40 + "\n")

    except Exception as e:
        logger.error(f"\n❌ Demonstration failed: {e}", exc_info=True)
        sys.exit(1)

