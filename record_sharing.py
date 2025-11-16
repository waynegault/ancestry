"""
Record Sharing Capabilities

Phase 5.5: Record Sharing Capabilities
Provides functionality to reference and share specific genealogical records in messages.

This module enables the assistant to reference specific records (birth, death, census, etc.)
when responding to DNA matches, making conversations more informative and helpful.
"""

from standard_imports import *
from test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


def format_record_reference(
    record_type: str,
    person_name: str,
    record_details: dict[str, str],
    include_source: bool = True
) -> str:
    """
    Format a genealogical record reference for inclusion in messages.

    Phase 5.5: Record Sharing Capabilities
    Creates formatted references to specific genealogical records.

    Args:
        record_type: Type of record (birth, death, census, marriage, etc.)
        person_name: Name of the person the record is about
        record_details: Dictionary with record details (date, place, source, etc.)
        include_source: Whether to include source citation

    Returns:
        Formatted string describing the record

    Example record_details:
        {
            'date': '1941',
            'place': 'Banff, Banffshire, Scotland',
            'source': '1939 Register',
            'url': 'https://www.ancestry.co.uk/...'
        }
    """
    record_type_lower = record_type.lower()

    # Build the basic record description
    parts = [f"{person_name}'s {record_type_lower} record"]

    # Add date if available
    if record_details.get('date'):
        parts.append(f"from {record_details['date']}")

    # Add place if available
    if record_details.get('place'):
        parts.append(f"in {record_details['place']}")

    description = " ".join(parts)

    # Add source citation if requested and available
    if include_source and 'source' in record_details and record_details['source']:
        description += f" (Source: {record_details['source']})"

    return description


def format_multiple_records(
    person_name: str,
    records: list[dict[str, Any]],
    max_records: int = 3
) -> str:
    """
    Format multiple record references for a person.

    Phase 5.5: Record Sharing Capabilities
    Creates a formatted list of multiple records for one person.

    Args:
        person_name: Name of the person
        records: List of record dictionaries, each with 'type' and 'details' keys
        max_records: Maximum number of records to include

    Returns:
        Formatted string listing the records

    Example records:
        [
            {
                'type': 'Birth',
                'details': {'date': '1941', 'place': 'Banff, Scotland', 'source': 'Birth Certificate'}
            },
            {
                'type': 'Census',
                'details': {'date': '1951', 'place': 'Aberdeen, Scotland', 'source': '1951 Census'}
            }
        ]
    """
    if not records:
        return f"No records found for {person_name}."

    # Limit to max_records
    records_to_show = records[:max_records]

    # Format each record
    formatted_records = []
    for record in records_to_show:
        record_type = record.get('type', 'Unknown')
        details = record.get('details', {})
        formatted = format_record_reference(record_type, person_name, details)
        formatted_records.append(f"• {formatted}")

    # Build the result
    result = f"Records for {person_name}:\n" + "\n".join(formatted_records)

    # Add note if there are more records
    if len(records) > max_records:
        remaining = len(records) - max_records
        result += f"\n• ... and {remaining} more record{'s' if remaining > 1 else ''}"

    return result


def create_record_sharing_message(
    person_name: str,
    records: list[dict[str, Any]],
    context: str = ""
) -> str:
    """
    Create a complete message about shared records.

    Phase 5.5: Record Sharing Capabilities
    Creates a full message for sharing records with DNA matches.

    Args:
        person_name: Name of the person
        records: List of record dictionaries
        context: Optional context to add before the records

    Returns:
        Complete message text
    """
    message_parts = []

    # Add context if provided
    if context:
        message_parts.append(context)
        message_parts.append("")  # Blank line

    # Add records
    records_text = format_multiple_records(person_name, records)
    message_parts.append(records_text)

    return "\n".join(message_parts)


def extract_record_url(record_details: dict[str, str]) -> Optional[str]:
    """
    Extract URL from record details if available.

    Phase 5.5: Record Sharing Capabilities
    Extracts and validates record URLs.

    Args:
        record_details: Dictionary with record details

    Returns:
        URL string if available, None otherwise
    """
    url = record_details.get('url')
    if url and url.startswith('http'):
        return url
    return None


def format_record_with_link(
    record_type: str,
    person_name: str,
    record_details: dict[str, str]
) -> str:
    """
    Format a record reference with clickable link if available.

    Phase 5.5: Record Sharing Capabilities
    Creates formatted record reference with optional URL.

    Args:
        record_type: Type of record
        person_name: Name of the person
        record_details: Dictionary with record details including optional 'url'

    Returns:
        Formatted string with link if available
    """
    # Get basic formatted reference
    basic_ref = format_record_reference(record_type, person_name, record_details)

    # Add link if available
    url = extract_record_url(record_details)
    if url:
        return f"{basic_ref}\nView record: {url}"

    return basic_ref


# ==============================================
# TESTS
# ==============================================


def test_basic_record_formatting():
    """Test basic record reference formatting."""
    details = {
        'date': '1941',
        'place': 'Banff, Banffshire, Scotland',
        'source': 'Birth Certificate'
    }

    result = format_record_reference('Birth', 'Fraser Gault', details)

    assert 'Fraser Gault' in result, "Should include person name"
    assert 'birth record' in result.lower(), "Should include record type"
    assert '1941' in result, "Should include date"
    assert 'Banff' in result, "Should include place"
    assert 'Birth Certificate' in result, "Should include source"


def test_record_without_source():
    """Test record formatting without source citation."""
    details = {
        'date': '1941',
        'place': 'Banff, Scotland',
        'source': 'Birth Certificate'
    }

    result = format_record_reference('Birth', 'Fraser Gault', details, include_source=False)

    assert 'Fraser Gault' in result, "Should include person name"
    assert 'Birth Certificate' not in result, "Should not include source when disabled"


def test_multiple_records_formatting():
    """Test formatting multiple records."""
    records = [
        {
            'type': 'Birth',
            'details': {'date': '1941', 'place': 'Banff, Scotland', 'source': 'Birth Certificate'}
        },
        {
            'type': 'Census',
            'details': {'date': '1951', 'place': 'Aberdeen, Scotland', 'source': '1951 Census'}
        },
        {
            'type': 'Death',
            'details': {'date': '2020', 'place': 'Edinburgh, Scotland', 'source': 'Death Certificate'}
        }
    ]

    result = format_multiple_records('Fraser Gault', records, max_records=2)

    assert 'Fraser Gault' in result, "Should include person name"
    assert 'Birth' in result, "Should include first record"
    assert 'Census' in result, "Should include second record"
    assert 'Death' not in result, "Should not include third record (max=2)"
    assert '1 more record' in result, "Should indicate remaining records"


def test_record_with_url():
    """Test record formatting with URL."""
    details = {
        'date': '1941',
        'place': 'Banff, Scotland',
        'source': 'Birth Certificate',
        'url': 'https://www.ancestry.co.uk/records/12345'
    }

    result = format_record_with_link('Birth', 'Fraser Gault', details)

    assert 'Fraser Gault' in result, "Should include person name"
    assert 'https://www.ancestry.co.uk' in result, "Should include URL"
    assert 'View record:' in result, "Should include link label"


def test_complete_sharing_message():
    """Test complete record sharing message creation."""
    records = [
        {
            'type': 'Birth',
            'details': {'date': '1941', 'place': 'Banff, Scotland'}
        }
    ]

    context = "I found some records that might help with your research:"
    result = create_record_sharing_message('Fraser Gault', records, context)

    assert context in result, "Should include context"
    assert 'Fraser Gault' in result, "Should include person name"
    assert 'birth record' in result.lower(), "Should include record type"


def test_url_extraction():
    """Test URL extraction from record details."""
    # Valid URL
    details_valid = {'url': 'https://www.ancestry.co.uk/records/12345'}
    url_valid = extract_record_url(details_valid)
    assert url_valid == 'https://www.ancestry.co.uk/records/12345', "Should extract valid URL"

    # No URL
    details_none = {'date': '1941'}
    url_none = extract_record_url(details_none)
    assert url_none is None, "Should return None when no URL"

    # Invalid URL
    details_invalid = {'url': 'not-a-url'}
    url_invalid = extract_record_url(details_invalid)
    assert url_invalid is None, "Should return None for invalid URL"


def _test_minimal_record_formatting():
    """Test record formatting with minimal details."""
    formatted = format_record_reference(
        "Birth",
        "John Smith",
        {},
        include_source=False
    )

    assert "John Smith" in formatted, "Should include person name"
    assert "birth" in formatted.lower(), "Should include record type"

    logger.info("✓ Minimal record formatting works correctly")
    return True


def _test_record_with_date_only():
    """Test record formatting with only date."""
    formatted = format_record_reference(
        "Death",
        "Mary Brown",
        {'date': '1920'},
        include_source=False
    )

    assert "Mary Brown" in formatted, "Should include person name"
    assert "1920" in formatted, "Should include date"
    assert "death" in formatted.lower(), "Should include record type"

    logger.info("✓ Record with date only formatted correctly")
    return True


def _test_record_with_place_only():
    """Test record formatting with only place."""
    formatted = format_record_reference(
        "Census",
        "William Gault",
        {'place': 'Aberdeen, Scotland'},
        include_source=False
    )

    assert "William Gault" in formatted, "Should include person name"
    assert "Aberdeen, Scotland" in formatted, "Should include place"
    assert "census" in formatted.lower(), "Should include record type"

    logger.info("✓ Record with place only formatted correctly")
    return True


def _test_empty_multiple_records():
    """Test multiple records formatting with empty list."""
    formatted = format_multiple_records("John Smith", [])

    assert isinstance(formatted, str), "Should return string"
    assert "John Smith" in formatted, "Should include person name"
    assert "no records" in formatted.lower(), "Should indicate no records"

    logger.info("✓ Empty multiple records handled correctly")
    return True


def _test_single_record_in_list():
    """Test multiple records formatting with single record."""
    records = [
        {
            'type': 'Birth',
            'details': {'date': '1850', 'place': 'Scotland'}
        }
    ]

    formatted = format_multiple_records("John Smith", records)

    assert "John Smith" in formatted, "Should include person name"
    assert "1850" in formatted, "Should include date"
    assert "Scotland" in formatted, "Should include place"

    logger.info("✓ Single record in list formatted correctly")
    return True


def _test_record_url_extraction_missing():
    """Test URL extraction with missing URL."""
    url = extract_record_url({})

    assert url is None, "Should return None for missing URL"

    logger.info("✓ Missing URL extraction handled correctly")
    return True


def _test_record_url_extraction_present():
    """Test URL extraction with present URL."""
    test_url = "https://www.ancestry.co.uk/test"
    url = extract_record_url({'url': test_url})

    assert url == test_url, "Should return the URL"

    logger.info("✓ URL extraction works correctly")
    return True


def _test_record_with_link_no_url():
    """Test record formatting with link when no URL present."""
    formatted = format_record_with_link(
        "Birth",
        "Jane Doe",
        {'date': '1900'}
    )

    assert "Jane Doe" in formatted, "Should include person name"
    assert "1900" in formatted, "Should include date"

    logger.info("✓ Record with link (no URL) formatted correctly")
    return True


def _test_sharing_message_minimal():
    """Test sharing message creation with minimal parameters."""
    message = create_record_sharing_message(
        "John Smith",
        []
    )

    assert isinstance(message, str), "Should return string"
    assert "John Smith" in message, "Should include person name"

    logger.info("✓ Minimal sharing message created correctly")
    return True


def record_sharing_module_tests() -> bool:
    """Run all record sharing tests."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Record Sharing Capabilities", __name__)
    suite.start_suite()

    with suppress_logging():
        # Edge case tests
        suite.run_test(
            "Minimal record formatting",
            _test_minimal_record_formatting,
            "Minimal record formatting works correctly",
            "Test record formatting with minimal details",
            "Verify formatting with no date, place, or source"
        )

        suite.run_test(
            "Record with date only",
            _test_record_with_date_only,
            "Record with date only formatted correctly",
            "Test record formatting with only date",
            "Verify formatting with date but no place or source"
        )

        suite.run_test(
            "Record with place only",
            _test_record_with_place_only,
            "Record with place only formatted correctly",
            "Test record formatting with only place",
            "Verify formatting with place but no date or source"
        )

        suite.run_test(
            "Empty multiple records",
            _test_empty_multiple_records,
            "Empty multiple records handled correctly",
            "Test multiple records formatting with empty list",
            "Verify graceful handling of empty record list"
        )

        suite.run_test(
            "Single record in list",
            _test_single_record_in_list,
            "Single record in list formatted correctly",
            "Test multiple records formatting with single record",
            "Verify formatting with one record in list"
        )

        suite.run_test(
            "URL extraction missing",
            _test_record_url_extraction_missing,
            "Missing URL extraction handled correctly",
            "Test URL extraction with missing URL",
            "Verify None returned for missing URL"
        )

        suite.run_test(
            "URL extraction present",
            _test_record_url_extraction_present,
            "URL extraction works correctly",
            "Test URL extraction with present URL",
            "Verify URL returned correctly"
        )

        suite.run_test(
            "Record with link no URL",
            _test_record_with_link_no_url,
            "Record with link (no URL) formatted correctly",
            "Test record formatting with link when no URL present",
            "Verify formatting without URL"
        )

        suite.run_test(
            "Sharing message minimal",
            _test_sharing_message_minimal,
            "Minimal sharing message created correctly",
            "Test sharing message creation with minimal parameters",
            "Verify message with empty records list"
        )

        # Original tests
        suite.run_test(
            "Basic record reference formatting",
            test_basic_record_formatting,
            "Record references formatted correctly with all details",
            "Test basic record formatting",
            "Testing record formatting with date, place, and source"
        )

        suite.run_test(
            "Record formatting without source",
            test_record_without_source,
            "Record formatted correctly without source citation",
            "Test record formatting without source",
            "Testing optional source citation parameter"
        )

        suite.run_test(
            "Multiple records formatting",
            test_multiple_records_formatting,
            "Multiple records formatted correctly with max limit",
            "Test multiple record formatting",
            "Testing record list with max_records limit"
        )

        suite.run_test(
            "Record with URL link",
            test_record_with_url,
            "Record formatted correctly with clickable URL",
            "Test record formatting with URL",
            "Testing URL inclusion in record references"
        )

        suite.run_test(
            "Complete sharing message",
            test_complete_sharing_message,
            "Complete message created with context and records",
            "Test complete message creation",
            "Testing full message generation with context"
        )

        suite.run_test(
            "URL extraction",
            test_url_extraction,
            "URLs extracted and validated correctly",
            "Test URL extraction",
            "Testing URL extraction with valid/invalid/missing URLs"
        )

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
run_comprehensive_tests = create_standard_test_runner(record_sharing_module_tests)


if __name__ == "__main__":
    print("🤖 Running Record Sharing Capabilities comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

