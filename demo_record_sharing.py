"""
Demonstration: Record Sharing Capabilities

Phase 5.5: Record Sharing Capabilities
Demonstrates the creation of formatted record references for genealogical messages.

This script shows how record references can be formatted and shared in messages
to DNA matches, making conversations more informative and helpful.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging

from record_sharing import (
    create_record_sharing_message,
    format_multiple_records,
    format_record_reference,
    format_record_with_link,
)
from standard_imports import *

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def demonstrate_single_record():
    """Demonstrate single record reference formatting."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 1: Single Record Reference")
    logger.info("="*80)

    # Example: Birth record
    birth_details = {
        'date': '1941',
        'place': 'Banff, Banffshire, Scotland',
        'source': 'Scotland Birth Certificate'
    }

    logger.info("\nFormatting a birth record:\n")
    birth_ref = format_record_reference('Birth', 'Fraser Gault', birth_details)
    logger.info(f"  {birth_ref}")

    # Example: Census record
    census_details = {
        'date': '1951',
        'place': 'Aberdeen, Aberdeenshire, Scotland',
        'source': '1951 Scotland Census'
    }

    logger.info("\nFormatting a census record:\n")
    census_ref = format_record_reference('Census', 'Fraser Gault', census_details)
    logger.info(f"  {census_ref}")

    logger.info("\n" + "="*80)


def demonstrate_record_with_url():
    """Demonstrate record reference with URL."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 2: Record Reference with URL")
    logger.info("="*80)

    # Example: Birth record with URL
    birth_details = {
        'date': '1941',
        'place': 'Banff, Banffshire, Scotland',
        'source': 'Scotland Birth Certificate',
        'url': 'https://www.ancestry.co.uk/discoveryui-content/view/12345:1234'
    }

    logger.info("\nFormatting a birth record with clickable URL:\n")
    birth_ref = format_record_with_link('Birth', 'Fraser Gault', birth_details)
    logger.info(f"  {birth_ref}")

    logger.info("\n" + "="*80)


def demonstrate_multiple_records():
    """Demonstrate multiple record references."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 3: Multiple Record References")
    logger.info("="*80)

    # Example: Multiple records for one person
    records = [
        {
            'type': 'Birth',
            'details': {
                'date': '1941',
                'place': 'Banff, Banffshire, Scotland',
                'source': 'Scotland Birth Certificate'
            }
        },
        {
            'type': 'Census',
            'details': {
                'date': '1951',
                'place': 'Aberdeen, Aberdeenshire, Scotland',
                'source': '1951 Scotland Census'
            }
        },
        {
            'type': 'Marriage',
            'details': {
                'date': '1965',
                'place': 'Edinburgh, Midlothian, Scotland',
                'source': 'Scotland Marriage Certificate'
            }
        },
        {
            'type': 'Census',
            'details': {
                'date': '1971',
                'place': 'Edinburgh, Midlothian, Scotland',
                'source': '1971 Scotland Census'
            }
        }
    ]

    logger.info("\nFormatting 4 records (showing max 3):\n")
    records_text = format_multiple_records('Fraser Gault', records, max_records=3)
    logger.info(records_text)

    logger.info("\n" + "="*80)


def demonstrate_complete_message():
    """Demonstrate complete record sharing message."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 4: Complete Record Sharing Message")
    logger.info("="*80)

    # Example: Complete message with context
    records = [
        {
            'type': 'Birth',
            'details': {
                'date': '1941',
                'place': 'Banff, Banffshire, Scotland',
                'source': 'Scotland Birth Certificate'
            }
        },
        {
            'type': 'Census',
            'details': {
                'date': '1951',
                'place': 'Aberdeen, Aberdeenshire, Scotland',
                'source': '1951 Scotland Census'
            }
        }
    ]

    context = "I found some records for Fraser Gault that might help with your research:"

    logger.info("\nComplete message with context and records:\n")
    logger.info("-" * 80)
    message = create_record_sharing_message('Fraser Gault', records, context)
    logger.info(message)
    logger.info("-" * 80)

    logger.info("\n" + "="*80)


def demonstrate_real_world_scenario():
    """Demonstrate real-world scenario with DNA match."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 5: Real-World DNA Match Scenario")
    logger.info("="*80)

    # Example: Responding to a DNA match about a common ancestor
    records = [
        {
            'type': 'Birth',
            'details': {
                'date': '1875',
                'place': 'Banff, Banffshire, Scotland',
                'source': 'Scotland Birth Certificate',
                'url': 'https://www.ancestry.co.uk/discoveryui-content/view/12345:1234'
            }
        },
        {
            'type': 'Census',
            'details': {
                'date': '1881',
                'place': 'Banff, Banffshire, Scotland',
                'source': '1881 Scotland Census',
                'url': 'https://www.ancestry.co.uk/discoveryui-content/view/23456:2345'
            }
        },
        {
            'type': 'Marriage',
            'details': {
                'date': '1895',
                'place': 'Aberdeen, Aberdeenshire, Scotland',
                'source': 'Scotland Marriage Certificate',
                'url': 'https://www.ancestry.co.uk/discoveryui-content/view/34567:3456'
            }
        }
    ]

    context = """Hi! I believe we share a common ancestor: John Gault (1875-1945).

I've found several records that confirm his details:"""

    logger.info("\nExample message to DNA match:\n")
    logger.info("=" * 80)
    message = create_record_sharing_message('John Gault', records, context)
    logger.info(message)

    # Add URLs separately for clarity
    logger.info("\n\nRecord Links:")
    for i, record in enumerate(records, 1):
        url = record['details'].get('url')
        if url:
            logger.info(f"  {i}. {record['type']}: {url}")

    logger.info("=" * 80)

    logger.info("\n" + "="*80)


def demonstrate_different_record_types():
    """Demonstrate different types of genealogical records."""
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION 6: Different Record Types")
    logger.info("="*80)

    record_types = [
        ('Birth', {'date': '1941', 'place': 'Banff, Scotland', 'source': 'Birth Certificate'}),
        ('Death', {'date': '2020', 'place': 'Edinburgh, Scotland', 'source': 'Death Certificate'}),
        ('Census', {'date': '1951', 'place': 'Aberdeen, Scotland', 'source': '1951 Census'}),
        ('Marriage', {'date': '1965', 'place': 'Edinburgh, Scotland', 'source': 'Marriage Certificate'}),
        ('Military', {'date': '1960', 'place': 'UK', 'source': 'Military Service Record'}),
        ('Immigration', {'date': '1950', 'place': 'Southampton, England', 'source': 'Passenger List'})
    ]

    logger.info("\nFormatting different record types:\n")
    for record_type, details in record_types:
        ref = format_record_reference(record_type, 'Fraser Gault', details)
        logger.info(f"  • {ref}")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    """Run all demonstrations."""
    logger.info("\n" + "🎯 "*40)
    logger.info("RECORD SHARING CAPABILITIES DEMONSTRATION")
    logger.info("Phase 5.5: Genealogical Record References for Messages")
    logger.info("🎯 "*40)

    try:
        # Run demonstrations
        demonstrate_single_record()
        demonstrate_record_with_url()
        demonstrate_multiple_records()
        demonstrate_complete_message()
        demonstrate_real_world_scenario()
        demonstrate_different_record_types()

        logger.info("\n" + "🎉 "*40)
        logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("🎉 "*40 + "\n")

    except Exception as e:
        logger.error(f"\n❌ Demonstration failed: {e}", exc_info=True)
        sys.exit(1)

