"""
Demonstration: Relationship Diagram Generation

Phase 5.4: Relationship Diagram Generation
Demonstrates the creation of ASCII/text-based relationship diagrams.

This script shows how relationship diagrams can be generated in different styles
for use in genealogical messages and responses.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from relationship_diagram import (
    format_relationship_for_message,
    generate_relationship_diagram,
)
from standard_imports import *

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def demonstrate_vertical_diagram():
    """Demonstrate vertical relationship diagram."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 1: Vertical Relationship Diagram")
    logger.info("="*80)

    # Example: Wayne → Fraser → John (grandfather)
    relationship_path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'John Gault', 'relationship': 'grandfather'},
        {'name': 'William Gault', 'relationship': 'great-grandfather'}
    ]

    logger.info("\nGenerating vertical diagram for 4-generation path:\n")
    diagram = generate_relationship_diagram(
        "Wayne Gault",
        "William Gault",
        relationship_path,
        "vertical"
    )

    logger.info(diagram)
    logger.info("\n" + "="*80)


def demonstrate_horizontal_diagram():
    """Demonstrate horizontal relationship diagram."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 2: Horizontal Relationship Diagram")
    logger.info("="*80)

    # Example: Wayne → Fraser → Mary (grandmother) → Jane (aunt) → Bob (1st cousin)
    relationship_path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'Mary Smith', 'relationship': 'grandmother'},
        {'name': 'Jane Smith', 'relationship': 'aunt'},
        {'name': 'Bob Smith', 'relationship': '1st cousin'}
    ]

    logger.info("\nGenerating horizontal diagram for 1st cousin relationship:\n")
    diagram = generate_relationship_diagram(
        "Wayne Gault",
        "Bob Smith",
        relationship_path,
        "horizontal"
    )

    logger.info(diagram)
    logger.info("\n" + "="*80)


def demonstrate_compact_diagram():
    """Demonstrate compact relationship diagram."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 3: Compact Relationship Diagram")
    logger.info("="*80)

    # Example: Long path for distant cousin
    relationship_path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'John Gault', 'relationship': 'grandfather'},
        {'name': 'William Gault', 'relationship': 'great-grandfather'},
        {'name': 'James Gault', 'relationship': '2x great-grandfather'},
        {'name': 'Robert Gault', 'relationship': '3x great-grandfather'},
        {'name': 'Thomas Gault', 'relationship': '4x great-grandfather'},
        {'name': 'Mary Brown', 'relationship': '4x great-grandmother'},
        {'name': 'Elizabeth Brown', 'relationship': '3x great-aunt'},
        {'name': 'Sarah Johnson', 'relationship': '2x great-cousin'},
        {'name': 'Michael Johnson', 'relationship': '1x great-cousin'},
        {'name': 'David Johnson', 'relationship': '4th cousin'}
    ]

    logger.info("\nGenerating compact diagram for distant cousin (12 generations):\n")
    diagram = generate_relationship_diagram(
        "Wayne Gault",
        "David Johnson",
        relationship_path,
        "compact"
    )

    logger.info(diagram)
    logger.info("\n" + "="*80)


def demonstrate_message_formatting():
    """Demonstrate relationship formatting for messages."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 4: Message Formatting with Relationship Diagrams")
    logger.info("="*80)

    # Example 1: Simple relationship
    simple_path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'}
    ]

    logger.info("\nExample 1: Simple relationship (father)\n")
    logger.info("With diagram:")
    logger.info("-" * 40)
    message_with = format_relationship_for_message(
        "Wayne Gault",
        "Fraser Gault",
        simple_path,
        include_diagram=True
    )
    logger.info(message_with)

    logger.info("\n\nWithout diagram:")
    logger.info("-" * 40)
    message_without = format_relationship_for_message(
        "Wayne Gault",
        "Fraser Gault",
        simple_path,
        include_diagram=False
    )
    logger.info(message_without)

    # Example 2: Complex relationship
    complex_path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'Mary Smith', 'relationship': 'grandmother'},
        {'name': 'Jane Smith', 'relationship': 'aunt'},
        {'name': 'Bob Smith', 'relationship': '1st cousin'}
    ]

    logger.info("\n\nExample 2: Complex relationship (1st cousin)\n")
    logger.info("With diagram:")
    logger.info("-" * 40)
    message_complex = format_relationship_for_message(
        "Wayne Gault",
        "Bob Smith",
        complex_path,
        include_diagram=True
    )
    logger.info(message_complex)

    logger.info("\n" + "="*80)


def demonstrate_all_styles_comparison():
    """Demonstrate all diagram styles side-by-side."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 5: Comparison of All Diagram Styles")
    logger.info("="*80)

    relationship_path = [
        {'name': 'Wayne Gault', 'relationship': 'self'},
        {'name': 'Fraser Gault', 'relationship': 'father'},
        {'name': 'John Gault', 'relationship': 'grandfather'},
        {'name': 'William Gault', 'relationship': 'great-grandfather'}
    ]

    logger.info("\nSame relationship shown in all three styles:\n")

    logger.info("VERTICAL STYLE:")
    logger.info("-" * 40)
    vertical = generate_relationship_diagram(
        "Wayne Gault",
        "William Gault",
        relationship_path,
        "vertical"
    )
    logger.info(vertical)

    logger.info("\n\nHORIZONTAL STYLE:")
    logger.info("-" * 40)
    horizontal = generate_relationship_diagram(
        "Wayne Gault",
        "William Gault",
        relationship_path,
        "horizontal"
    )
    logger.info(horizontal)

    logger.info("\n\nCOMPACT STYLE:")
    logger.info("-" * 40)
    compact = generate_relationship_diagram(
        "Wayne Gault",
        "William Gault",
        relationship_path,
        "compact"
    )
    logger.info(compact)

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    """Run all demonstrations."""
    logger.info("\n" + "🎯 "*40)
    logger.info("RELATIONSHIP DIAGRAM GENERATION DEMONSTRATION")
    logger.info("Phase 5.4: ASCII/Text-Based Genealogical Diagrams")
    logger.info("🎯 "*40)

    try:
        # Run demonstrations
        demonstrate_vertical_diagram()
        demonstrate_horizontal_diagram()
        demonstrate_compact_diagram()
        demonstrate_message_formatting()
        demonstrate_all_styles_comparison()

        logger.info("\n" + "🎉 "*40)
        logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("🎉 "*40 + "\n")

    except Exception as e:
        logger.error(f"\n❌ Demonstration failed: {e}", exc_info=True)
        sys.exit(1)

